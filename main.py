from os import path

from Solution import Board, Futoshiki_Solver

# Naama Omer 207644014, Moshe Zeev Hefter 205381379
if __name__ == '__main__':
    PATH_TO_TXT_FILE = input(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n             HELLO!\n    WELCOME TO FUTOSHIKI SOLVER!\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "Please enter PATH to a .txt file, with a list of paths"
        "to Futoshiki boards to solve\n"
        "An example of the outer .txt file:\n\n"
        "./inputs/5_easy.txt\n./inputs/5_tricky.txt\n./inputs/6_easy.txt"
        "\n./inputs/6_tricky.txt\n./inputs/7_easy.txt\n./inputs/7_tricky.txt\n\nPath:")

    assert path.exists(PATH_TO_TXT_FILE), "Path don't exist"
    assert path.isfile(PATH_TO_TXT_FILE), "Path dose not directed to file"
    assert PATH_TO_TXT_FILE.endswith('.txt'), "path is not a text type file"
    alg_code = input("Please press 1 - For regular evolutionary algorithm.\n"
                     "Please press 2 - For Darwin strategy with the evolutionary algorithm.\n"
                     "Please press 3 - For Lamarck strategy with the evolutionary algorithm.\n"
                     "Answer:\n")

    with open(PATH_TO_TXT_FILE, 'r') as fp:
        count = 0
        # colors = ['r', 'g', 'b', 'y', 'hotpink', 'c']
        print("\nLet's go!")
        for line in fp:
            line = line[:-1]
            print("Working on file " + str(count + 1) + "# in path: '" + line[:-1] + "' ...")
            iter_array = []
            average_score_array = []
            best_score_array = []
            board_current_level = Board(line)
            a = Futoshiki_Solver(board_current_level)

            if alg_code == '1':
                a.selection()
                i = 0
                while not a.evaluations[0] == 0 or i == 0:
                    a.selection()
                    if a.evaluations[0] == 0:
                        if i == 0:
                            print("Total calls to evaluation function: ", i, ", Best score: ", a.evaluations[0],
                                  ", Average score: ", sum(a.evaluations) / 100)
                            best_score_array.append(a.evaluations[0])
                            average_score_array.append(sum(a.evaluations) / 100)
                            iter_array.append(i)

                        print("Total calls to evaluation function: ", i, ", Best score: ", a.evaluations[0],
                              ", Average score: ", sum(a.evaluations) / 100)
                        best_score_array.append(a.evaluations[0])
                        average_score_array.append(sum(a.evaluations) / 100)
                        iter_array.append(i)
                        print("\nWe have reached a solution!\n")
                        print(a.solutions[0].board)
                        break

                    if i < 1000:
                        a.mutation()
                    else:
                        a.mutation(p=0.02)

                    if i % 100 == 0:
                        print("Generation number: ", i, ", Best score: ", a.evaluations[0],
                              ", Average score: ", sum(a.evaluations) / 100)
                        best_score_array.append(a.evaluations[0])
                        average_score_array.append(sum(a.evaluations) / 100)
                        iter_array.append(i)

                    if i >= 5000:
                        print("\nSorry, solution not found :(\nThis is the best solution after 5,000 generations:\n")
                        print(a.solutions[0].board)
                        break
                    i += 1
                    count += 1

            elif alg_code == '2':
                a.selection_Darwin()
                i = 0
                while not a.evaluations[0] == 0 or i == 0:
                    a.selection_Darwin()
                    if a.evaluations[0] == 0:
                        if i == 0:
                            print("Total calls to evaluation function: ", i, ", Best score: ", a.evaluations[0],
                                  ", Average score: ", sum(a.evaluations) / 100)
                            best_score_array.append(a.evaluations[0])
                            average_score_array.append(sum(a.evaluations) / 100)
                            iter_array.append(i)

                        print("Total calls to evaluation function: ", i, ", Best score: ", a.evaluations[0],
                              ", Average score: ", sum(a.evaluations) / 100)
                        best_score_array.append(a.evaluations[0])
                        average_score_array.append(sum(a.evaluations) / 100)
                        iter_array.append(i)
                        print("\nWe have reached a solution!\n")
                        print(a.solutions[0].board)
                        break

                    if i < 1000:
                        a.mutation()
                    else:
                        a.mutation(p=0.02)

                    if i % 100 == 0:
                        print("Generation number: ", i, ", Best score: ", a.evaluations[0],
                              ", Average score: ", sum(a.evaluations) / 100)
                        best_score_array.append(a.evaluations[0])
                        average_score_array.append(sum(a.evaluations) / 100)
                        iter_array.append(i)

                    if i >= 5000:
                        print("\nWe have not reached a solution this time :( try later!\n")
                        break
                    i += 1
                    count += 1

            elif alg_code == '3':
                a.selection_Lamarck()
                i = 0
                while not a.evaluations[0] == 0 or i == 0:
                    a.selection_Lamarck()
                    if a.evaluations[0] == 0:
                        if i == 0:
                            print("Total calls to evaluation function: ", i, ", Best score: ", a.evaluations[0],
                                  ", Average score: ", sum(a.evaluations) / 100)
                            best_score_array.append(a.evaluations[0])
                            average_score_array.append(sum(a.evaluations) / 100)
                            iter_array.append(i)

                        print("Total calls to evaluation function: ", i, ", Best score: ", a.evaluations[0],
                              ", Average score: ", sum(a.evaluations) / 100)
                        best_score_array.append(a.evaluations[0])
                        average_score_array.append(sum(a.evaluations) / 100)
                        iter_array.append(i)
                        print("\nWe have reached a solution!\n")
                        print(a.solutions[0].board)
                        break

                    if i < 1000:
                        a.mutation()
                    else:
                        a.mutation(p=0.02)

                    if i % 100 == 0:
                        print("Generation number: ", i, ", Best score: ", a.evaluations[0],
                              ", Average score: ", sum(a.evaluations) / 100)
                        best_score_array.append(a.evaluations[0])
                        average_score_array.append(sum(a.evaluations) / 100)
                        iter_array.append(i)

                    if i >= 5000:
                        print("\nWe have not reached a solution this time :( try later!\n")
                        break
                    i += 1
                    count += 1

            else:
                raise AssertionError(alg_code)

    print("Finished with " + alg_code + "for all files received.")
