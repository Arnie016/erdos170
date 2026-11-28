import Foundation
import Metal

struct AuditError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

struct Outcome {
    var pairCount: Int
    var freshTriples: Int
    var lostLabels: Int
    var lossHash: UInt64
    var deficit: Int
    var left: Int
    var right: Int
    var insertion: Int
}

func gapRuler(_ r: Int) -> [Int] {
    var marks = [0]
    for (count, gap) in [(r, 1), (1, r + 1), (r, 2*r + 1),
                         (2*r, 4*r + 3), (r + 1, 2*r + 2), (r, 1)] {
        for _ in 0..<count { marks.append(marks[marks.count - 1] + gap) }
    }
    return marks
}

func literalDeficit(_ marks: [Int], length: Int) -> Int {
    var seen = Set<Int>()
    for i in marks.indices {
        for j in (i + 1)..<marks.count { seen.insert(abs(marks[i] - marks[j])) }
    }
    return length - seen.count
}

struct MetalAudit {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let distancePipeline: MTLComputePipelineState
    let insertionPipeline: MTLComputePipelineState

    init(source: URL) throws {
        guard let gpu = MTLCreateSystemDefaultDevice(), let commands = gpu.makeCommandQueue() else {
            throw AuditError("Metal device or command queue unavailable")
        }
        let lib = try gpu.makeLibrary(source: String(contentsOf: source, encoding: .utf8), options: nil)
        guard let distance = lib.makeFunction(name: "missingDistances"),
              let insertion = lib.makeFunction(name: "bestInsertion") else {
            throw AuditError("Missing kernel in supplied Metal source")
        }
        device = gpu
        queue = commands
        distancePipeline = try gpu.makeComputePipelineState(function: distance)
        insertionPipeline = try gpu.makeComputePipelineState(function: insertion)
    }

    func arrayBuffer<T>(_ values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty,
              let result = values.withUnsafeBufferPointer({ ptr in
                  device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count * MemoryLayout<T>.stride,
                                    options: .storageModeShared)
              }) else { throw AuditError("Could not allocate Metal input buffer") }
        return result
    }

    func dispatch(_ pipeline: MTLComputePipelineState, buffers: [MTLBuffer],
                  length: Int, width: Int, rows: Int) throws {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw AuditError("Metal command encoding failed")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        var intervalLength = UInt32(length)
        encoder.setBytes(&intervalLength, length: MemoryLayout<UInt32>.stride, index: buffers.count)
        let groupWidth = pipeline.threadExecutionWidth
        let groupRows = min(4, pipeline.maxTotalThreadsPerThreadgroup / groupWidth)
        encoder.dispatchThreads(MTLSize(width: width, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: groupWidth, height: groupRows, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw AuditError("Metal command failed: \(String(describing: command.error))")
        }
    }

    func run(marks: [Int], length: Int) throws -> Outcome {
        guard marks == marks.sorted(), Set(marks).count == marks.count,
              marks.first == 0, marks.last == length, marks.count < length + 1,
              literalDeficit(marks, length: length) == 0 else {
            throw AuditError("Expected a complete, distinct, contained ruler with at least one fresh coordinate")
        }
        var labels = [UInt32](repeating: 0, count: length + 1)
        for (i, mark) in marks.enumerated() { labels[mark] = UInt32(i + 1) }
        var deletions: [SIMD2<UInt32>] = []
        var fibers = [[SIMD2<UInt32>]](repeating: [], count: length + 1)
        for i in marks.indices {
            for j in (i + 1)..<marks.count {
                let ids = SIMD2(UInt32(i + 1), UInt32(j + 1))
                deletions.append(ids)
                fibers[marks[j] - marks[i]].append(ids)
            }
        }
        var starts: [UInt32] = [0]
        var edges: [SIMD2<UInt32>] = []
        for fiber in fibers {
            edges.append(contentsOf: fiber)
            starts.append(UInt32(edges.count))
        }
        let deletedBuffer = try arrayBuffer(deletions)
        let flagCount = length * deletions.count
        guard let flagBuffer = device.makeBuffer(length: flagCount, options: .storageModeShared) else {
            throw AuditError("Cannot allocate distance flags")
        }
        try dispatch(distancePipeline, buffers: [deletedBuffer, try arrayBuffer(starts),
                     try arrayBuffer(edges), flagBuffer],
                     length: length, width: length, rows: deletions.count)
        let flags = flagBuffer.contents().bindMemory(to: UInt8.self, capacity: flagCount)
        var lostStarts: [UInt32] = [0]
        var lost: [UInt32] = []
        var digest: UInt64 = 14695981039346656037
        let prime: UInt64 = 1099511628211
        for p in deletions.indices {
            for d in stride(from: length, through: 1, by: -1) where flags[p * length + d - 1] != 0 {
                lost.append(UInt32(d))
                digest = (digest ^ UInt64(d)) &* prime
            }
            lostStarts.append(UInt32(lost.count))
            digest = digest &* prime // zero separates deletion pairs
        }
        let bestBuffer = try arrayBuffer([UInt32](repeating: UInt32.max, count: deletions.count))
        try dispatch(insertionPipeline, buffers: [deletedBuffer, try arrayBuffer(labels),
                     try arrayBuffer(lostStarts), try arrayBuffer(lost.isEmpty ? [0] : lost), bestBuffer],
                     length: length, width: length + 1, rows: deletions.count)
        let best = bestBuffer.contents().bindMemory(to: UInt32.self, capacity: deletions.count)
        var result = Outcome(pairCount: deletions.count, freshTriples: deletions.count * (length + 1 - marks.count),
                             lostLabels: lost.count, lossHash: digest, deficit: Int.max, left: 0, right: 0, insertion: 0)
        for p in deletions.indices {
            guard best[p] != UInt32.max else { throw AuditError("Missing result for deletion pair \(p)") }
            let deficit = Int(best[p]) / (length + 1)
            if deficit < result.deficit {
                result.deficit = deficit
                result.left = marks[Int(deletions[p].x) - 1]
                result.right = marks[Int(deletions[p].y) - 1]
                result.insertion = Int(best[p]) % (length + 1)
            }
        }
        let repaired = marks.filter { $0 != result.left && $0 != result.right } + [result.insertion]
        guard !marks.contains(result.insertion), (0...length).contains(result.insertion),
              Set(repaired).count == marks.count - 1,
              literalDeficit(repaired, length: length) == result.deficit else {
            throw AuditError("GPU minimizer failed literal CPU coverage")
        }
        return result
    }
}

do {
    let args = CommandLine.arguments
    guard args.count == 3, let maxR = Int(args[2]), (1...32).contains(maxR) else {
        throw AuditError("Usage: repair-gpu path/to/repair.metal MAX_R (1...32)")
    }
    let audit = try MetalAudit(source: URL(fileURLWithPath: args[1]))
    let control = try audit.run(marks: [0, 1, 2, 5, 6], length: 6)
    guard control.deficit == 0 else { throw AuditError("Positive repair control failed") }
    let negative = try audit.run(marks: [0, 1, 4, 6], length: 6)
    guard negative.deficit == 3 else { throw AuditError("Negative repair control failed") }
    let broken = [0, 1, 3, 6]
    guard literalDeficit(broken, length: 6) == 1 else { throw AuditError("Broken control failed") }
    FileHandle.standardError.write(Data("Device: \(audit.device.name)\nPASS positive repair, negative repair and broken coverage controls\n".utf8))
    print("r,N,marks,deletion_pairs,fresh_triples,lost_incidences,loss_hash,min_missing,delete_x,delete_y,insert_z")
    for r in 1...maxR {
        let marks = gapRuler(r), length = 12*r*r + 14*r + 3
        let row = try audit.run(marks: marks, length: length)
        print("\(r),\(length),\(marks.count),\(row.pairCount),\(row.freshTriples),\(row.lostLabels),\(row.lossHash),\(row.deficit),\(row.left),\(row.right),\(row.insertion)")
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
