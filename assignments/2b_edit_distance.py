"""SLP v3 2.4 - 2.7"""


def compute_ed(word1, word2):
    ed_table = [[0 for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
    ed_table[0][0] = 0
    trace_table = [[[] for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                ed_table[0][j] = j
                if j > 0:
                    trace_table[0][j].append(f"Insert {word2[j-1]}")
            elif j == 0:
                ed_table[i][0] = i
                if i > 0:
                    trace_table[i][0].append(f"Delete {word1[i-1]}")
            else:
                ed_table[i][j] = min(
                    ed_table[i - 1][j] + 1,
                    ed_table[i][j - 1] + 1,
                    ed_table[i - 1][j - 1] + (word1[i - 1] != word2[j - 1]),
                )
                if ed_table[i - 1][j] + 1 == ed_table[i][j]:
                    trace_table[i][j] = trace_table[i - 1][j] + [f"Delete {word1[i-1]}"]
                elif ed_table[i][j - 1] + 1 == ed_table[i][j]:
                    trace_table[i][j] = trace_table[i][j - 1] + [f"Insert {word2[j-1]}"]
                elif word1[i - 1] != word2[j - 1]:
                    trace_table[i][j] = trace_table[i - 1][j - 1] + [
                        f"Substitute {word1[i-1]} with {word2[j-1]}"
                    ]
                else:
                    trace_table[i][j] = trace_table[i - 1][j - 1]

    return ed_table[-1][-1], trace_table[-1][-1]


if __name__ == "__main__":
    print(compute_ed("leda", "deal"))
    print(compute_ed("drive", "brief"))
    print(compute_ed("drive", "divers"))
