from assignment2 import *
import datetime
data_folder = ""#"data_test\\"

'''
time=datetime.datetime.now()
list=['big.txt']
error_list=['data_test/commmon_errors.txt']
alex2=learn_language_model(list,n=3)
'''


def levenshteinDistance(a, b):
    # "Infinity" -- greater than maximum possible edit distance
    # Used to prevent transpositions for first characters
    INF = len(a) + len(b)

    # Matrix: (M + 2) x (N + 2)
    matrix  = [[INF for n in range(len(b) + 2)]]
    matrix += [[INF] + list(range(len(b) + 1))]
    matrix += [[INF, m] + [0] * len(b) for m in range(1, len(a) + 1)]

    # Holds last row each element was encountered: DA in the Wikipedia pseudocode
    last_row = {}

    # Fill in costs
    for row in range(1, len(a) + 1):
        # Current character in a
        ch_a = a[row-1]

        # Column of last match on this row: DB in pseudocode
        last_match_col = 0

        for col in range(1, len(b) + 1):
            # Current character in b
            ch_b = b[col-1]

            # Last row with matching character
            last_matching_row = last_row.get(ch_b, 0)

            # Cost of substitution
            cost = 0 if ch_a == ch_b else 1

            # Compute substring distance
            matrix[row+1][col+1] = min(
                matrix[row][col] + cost, # Substitution
                matrix[row+1][col] + 1,  # Addition
                matrix[row][col+1] + 1,  # Deletion

                # Transposition
                # Start by reverting to cost before transposition
                matrix[last_matching_row][last_match_col]
                    # Cost of letters between transposed letters
                    # 1 addition + 1 deletion = 1 substitution
                    + max((row - last_matching_row - 1),
                          (col - last_match_col - 1))
                    # Cost of the transposition itself
                    + 1)

            # If there was a match, update last_match_col
            if cost == 0:
                last_match_col = col

        # Update last row for current character
        last_row[ch_a] = row

    # Return last element
    return matrix[-1][-1]

error_list=['commmon_errors.txt']



alex=levenshteinDistance('xlea','alex')
print(alex)