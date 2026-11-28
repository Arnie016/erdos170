// Independent portable minimum-deficit calculation using literal surviving pairs.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Outcome {
    std::uint64_t pairs = 0, triples = 0, lost = 0;
    std::uint64_t hash = UINT64_C(14695981039346656037);
    int deficit = std::numeric_limits<int>::max(), left = -1, right = -1, z = -1;
};

std::vector<int> block_ruler(int r) {
    const int a = 2*r + 1, b = a + 1, q = a + b, start = (r+1)*a;
    const int H = (3*r+1)*q;
    std::vector<int> marks;
    for (int x = 0; x <= r; ++x) marks.push_back(x);
    for (int i = 1; i <= r; ++i) marks.push_back(i*a);
    for (int j = 0; j <= 2*r; ++j) marks.push_back(start+j*q);
    for (int i = r; i >= 1; --i) marks.push_back(H-i*b);
    for (int y = 0; y <= r; ++y) marks.push_back(H+y);
    return marks;
}

std::vector<unsigned char> coverage(const std::vector<int>& marks, int length) {
    std::vector<unsigned char> covered(length+1, 0);
    for (std::size_t j = 0; j < marks.size(); ++j) {
        for (std::size_t i = 0; i < j; ++i) {
            covered[std::abs(marks[i] - marks[j])] = 1;
        }
    }
    return covered;
}

Outcome solve(const std::vector<int>& marks, int length) {
    if (marks.empty() || marks.front() != 0 || marks.back() != length ||
        !std::is_sorted(marks.begin(), marks.end()) ||
        std::adjacent_find(marks.begin(), marks.end()) != marks.end()) {
        throw std::runtime_error("Invalid mark set");
    }
    const auto original_cover = coverage(marks, length);
    if (std::find(original_cover.begin()+1, original_cover.end(), 0) != original_cover.end()) {
        throw std::runtime_error("Incomplete baseline");
    }
    std::vector<unsigned char> original(length+1, 0);
    for (int x : marks) original[x] = 1;
    std::vector<int> fresh;
    for (int z = 0; z <= length; ++z) if (!original[z]) fresh.push_back(z);
    if (fresh.empty()) throw std::runtime_error("Empty insertion domain");
    Outcome result;
    constexpr std::uint64_t prime = UINT64_C(1099511628211);
    for (std::size_t i = 0; i < marks.size(); ++i) {
        for (std::size_t j = i+1; j < marks.size(); ++j) {
            ++result.pairs;
            result.triples += fresh.size();
            std::vector<int> kept;
            for (int x : marks) if (x != marks[i] && x != marks[j]) kept.push_back(x);
            auto covered = coverage(kept, length);
            std::vector<int> gain(length+1, 0), last(length+1, -1);
            int n_lost = 0;
            for (int d = length; d >= 1; --d) {
                if (covered[d]) continue;
                ++n_lost;
                result.hash = (result.hash ^ static_cast<std::uint64_t>(d)) * prime;
                for (int x : kept) {
                    // The same insertion can hit d from either side. Credit d once.
                    for (int z : {x-d, x+d}) {
                        if (z < 0 || z > length || original[z] || last[z] == d) continue;
                        last[z] = d;
                        ++gain[z];
                    }
                }
            }
            result.lost += n_lost;
            result.hash *= prime;
            for (int z : fresh) {
                int deficit = n_lost - gain[z];
                if (deficit < result.deficit) {
                    result.deficit = deficit;
                    result.left = marks[i]; result.right = marks[j]; result.z = z;
                }
            }
        }
    }
    std::vector<int> edited;
    for (int x : marks) if (x != result.left && x != result.right) edited.push_back(x);
    edited.push_back(result.z);
    auto direct = coverage(edited, length);
    if (std::count(direct.begin()+1, direct.end(), 0) != result.deficit) {
        throw std::runtime_error("Minimizer failed full pair enumeration");
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: verify-cpu MAX_R (1...32)");
        std::size_t consumed = 0;
        const int bound = std::stoi(argv[1], &consumed);
        if (consumed != std::string(argv[1]).size() || bound < 1 || bound > 32) {
            throw std::runtime_error("MAX_R must be an integer in 1...32");
        }
        if (solve({0,1,2,5,6}, 6).deficit != 0 || solve({0,1,4,6}, 6).deficit != 3) {
            throw std::runtime_error("Control failed");
        }
        std::cerr << "PASS positive and negative repair controls\n";
        std::cout << "r,N,marks,deletion_pairs,fresh_triples,lost_incidences,loss_hash,min_missing,delete_x,delete_y,insert_z\n";
        for (int r = 1; r <= bound; ++r) {
            const auto marks = block_ruler(r);
            const int length = (3*r+1)*(4*r+3) + r;
            const auto x = solve(marks, length);
            std::cout << r << ',' << length << ',' << marks.size() << ',' << x.pairs << ','
                      << x.triples << ',' << x.lost << ',' << x.hash << ',' << x.deficit << ','
                      << x.left << ',' << x.right << ',' << x.z << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
