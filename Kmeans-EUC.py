import math

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Constants
K = 2  # amount of clusters
DIM = 3  # points dimension
N = 5  # amount of points

# Data structures
points = [
    [1, 2, 3],  # P1
    [0, 1, 2],  # P2
    [3, 0, 5],  # P3
    [4, 1, 3],  # P4
    [5, 0, 1]  # P5
]

centroids = [
    [1, 0, 0],  # Initial C1
    [0, 1, 1]  # Initial C2
]

assignments = [0] * N


def euclidean_distance(a, b):
    calc_str = "√["
    sum_squares = 0

    for i in range(DIM):
        diff = a[i] - b[i]
        sum_squares += diff * diff
        #calc_str += f"({a[i]:.0f}-{b[i]:.2f})²"
        calc_str += f"({diff:.2f})²"
        if i != DIM - 1:
            calc_str += " + "

    result = math.sqrt(sum_squares)
    calc_str += f"] = {result:.2f}"

    return result, calc_str


def print_table(distances, calculations):
    print(
        "+----+-----------------------------------------------------+-----------------------------------------------------+--------+")
    print(
        "| Pt | Distance to C1                                      | Distance to C2                                      | Assign |")
    print(
        "+----+-----------------------------------------------------+-----------------------------------------------------+--------+")

    for i in range(N):
        d1 = distances[i][0]
        d2 = distances[i][1]
        assigned = 0 if d1 <= d2 else 1

        c1_str = calculations[i][0]
        c2_str = calculations[i][1]

        # Add asterisk to the smaller distance
        if assigned == 0:
            c1_str += " *"
            c2_str += " "
        else:
            c1_str += " "
            c2_str += " *"

        print(f"| P{i + 1:<2}| {c1_str:<52}| {c2_str:<52}|  C{assigned + 1}    |")

    print(
        "+----+-----------------------------------------------------+-----------------------------------------------------+--------+")


def print_assignments():
    for c in range(K):
        print(f"{CYAN}C{c + 1}: {{", end="")
        first = True
        for i in range(N):
            if assignments[i] == c:
                if not first:
                    print(", ", end="")
                print(f"P{i + 1}", end="")
                first = False
        print(f"}}{RESET}")



def print_new_centroids(new_centroids, counts):
    for c in range(K):
        print(f"{YELLOW}\nNew centroid C{c + 1} calculation:{RESET}")
        print("Coordinates: (", end="")

        for d in range(DIM):
            print("(", end="")
            first = True
            for i in range(N):
                if assignments[i] == c:
                    if not first:
                        print("+", end="")
                    print(f"{points[i][d]:.2f}", end="")
                    first = False
            print(f")/{counts[c]}", end="")
            if d != DIM - 1:
                print(", ", end="")
        print(")")

        print("Result: (", end="")
        for d in range(DIM):
            value = (new_centroids[c][d] / counts[c]) if counts[c] > 0 else 0
            print(f"{value:.2f}", end="")
            if d != DIM - 1:
                print(", ", end="")
        print(")")


def kmeans():
    changed = True
    iteration = 1

    while changed:
        print(f"{RED}\n{'=' * 36} ITERATION {iteration} {'=' * 36}{RESET}")
        changed = False

        # Calculate distances
        distances = [[0] * K for _ in range(N)]
        calculations = [[""] * K for _ in range(N)]

        for i in range(N):
            for c in range(K):
                distances[i][c], calculations[i][c] = euclidean_distance(points[i], centroids[c])

        print_table(distances, calculations)

        # Initialize for new centroids calculation
        counts = [0] * K
        new_centroids = [[0] * DIM for _ in range(K)]

        # Assign points to clusters
        for i in range(N):
            assigned = 0 if distances[i][0] <= distances[i][1] else 1
            if assignments[i] != assigned:
                changed = True
            assignments[i] = assigned

            # Add point coordinates to new centroid calculation
            for d in range(DIM):
                new_centroids[assigned][d] += points[i][d]
            counts[assigned] += 1

        print(f"{GREEN}\nAssignments after iteration {iteration}:{RESET}")
        print_assignments()

        if not changed:
            print(f"{RED}\nNo changes in assignments. Final clustering reached.{RESET}")
            break

        print_new_centroids(new_centroids, counts)

        # Update centroids
        for c in range(K):
            if counts[c] > 0:
                for d in range(DIM):
                    centroids[c][d] = new_centroids[c][d] / counts[c]

        iteration += 1


def main():
    print(f"{GREEN}Starting K-Means Clustering (Euclidean Distance)...{RESET}")
    kmeans()
    print(f"{YELLOW}\nFinished clustering!!!{RESET}")


if __name__ == "__main__":
    main()