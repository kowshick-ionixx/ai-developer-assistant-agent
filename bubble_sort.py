"""
Bubble sort implementation module.
"""


def bubble_sort(arr):
    """
    Sorts a list in ascending order using the bubble sort algorithm.
    """
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        swapped = False

        # Last i elements are already in place, so we don't need to check them
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no two elements were swapped in the inner loop, the list is sorted
        if not swapped:
            break

    return arr
