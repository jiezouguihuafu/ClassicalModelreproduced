# https://www.cnblogs.com/onepixel/articles/7674659.html

# TODO 1.冒泡排序
def bubble_sort(array):
    """
    每次对比两个元素，如果右边比左边大，就调换位置，逐渐把最大的元素移动到最右边，
    空间复杂度：  O(1)
    时间复杂度：  O(n2)
    稳定性：    稳定

    :param array:list
    :return:list
    """
    for i in range(len(array)):
        for j in range(len(array)-i-1):
            if array[j]>array[j+1]:
                array[j],array[j+1]=array[j+1],array[j]
        # print(array)
    return array


# TODO 2.选择排序
def select_sort(array):
    """
    首先在未排序序列中找到最小元素，存放到排序序列的起始位置
    然后，再从剩余未排序元素中继续寻找最小元素，然后放到已排序序列的末尾。
    以此类推，直到所有元素均排序完毕。

    空间复杂度：  O(1)
    时间复杂度：  O(n2)
    稳定性：    稳定

    :param array: list
    :return: list
    """
    for i in range(len(array)):
        min_index = i
        for j in range(i+1,len(array)):
            if array[j]<array[min_index]:
                min_index=j
        array[i],array[min_index] =array[min_index], array[i]
    return array


# TODO 3.插入排序
def insertion_sort(array):
    """
    每次用当前值和之前所有值对比，如果比前面小，就交换，然后再和前面对比，如果大，就终止

    空间复杂度：  O(1)
    时间复杂度：  O(n2)
    稳定性：    稳定

    :param array: list
    :return: list
    """
    for i in range(1,len(array)):
        j=i
        while j>0:
            if array[j]<array[j-1]:
                array[j],array[j-1]=array[j-1],array[j]
                j-=1
            else:
                break
    return array


# TODO 4.希尔排序
def shell_sort(arr):
    """
    插入排序对于大规模的乱序数组的时候效率是比较慢的，因为它每次只能将数据移动一位，
    希尔排序为了加快插入的速度，让数据移动的时候可以实现跳跃移动，节省了一部分的时间开支

    空间复杂度：  O(1)
    时间复杂度：  最差O(n2)，最好O(n)，平均O(nLog2n)
    稳定性：    不稳定

    :param arr:
    :return:
    """
    n = len(arr)
    gap = int(n / 2)

    while gap > 0:

        for i in range(gap, n):

            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap = int(gap / 2)
    return arr


def merge_sort_helper(left,right):
    new = []
    left_index = 0
    right_index = 0
    left_len = len(left)
    right_len = len(right)

    while left_index<left_len and right_index<right_len:
        if left[left_index]<right[right_index]:
            new.append(left[left_index])
            left_index+=1
        else:
            new.append(right[right_index])
            right_index+=1

    if left_index == left_len:
        new.extend(right[right_index:])
    else:
        new.extend(left[left_index:])
    return new


# TODO 5.归并排序
def merge_sort(array):
    """
    将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。
    若将两个有序表合并成一个有序表，称为2-路归并。

    空间复杂度：  O(n)
    时间复杂度：  O(nLog2n)
    稳定性：    不稳定

    :param array:
    :return:
    """
    if len(array)<=1:return array

    mid = len(array)//2
    left = merge_sort(array[:mid])
    right = merge_sort(array[mid:])
    return merge_sort_helper(left,right)


# TODO 6.快速排序
def quick_sort(array):
    """
    通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，
    则可分别对这两部分记录继续进行排序，以达到整个序列有序

    空间复杂度：  O(nLog2n)
    时间复杂度：  最差O(n2)，最好O(nLog2n)，平均O(nLog2n)
    稳定性：    不稳定

    :param array:
    :return:
    """
    if not array:return array
    first = array[0]
    left = []
    right = []
    for i in array[1:]:
        if i<first:
            left.append(i)
        else:
            right.append(i)

    return quick_sort(left) + [first] + quick_sort(right)





if __name__ == '__main__':
    arr = [9,1,5,2,6,0,8,11]
    print(quick_sort(arr))



