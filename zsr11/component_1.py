import numpy as np
import matplotlib
import math
import typing




def check_SOn(matrix, epsilon) -> bool:
    # get determinant
    if (len(matrix) == 2):
        determinant = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    elif (len(matrix) == 3):
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
    else:
        return False
    
    #check if its approx 1
    if abs(determinant - 1) > epsilon:
        return False
    
    #check orthongonality
    transposed = np.transpose(matrix)   
    identity = np.eye(len(matrix))
    result = np.dot(transposed, matrix) 

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if abs(result[i][j] - identity[i][j]) > epsilon:
                return False
            
    return True

def check_quaternion(vector, epsilon) -> bool:
    if len(vector) != 4:
        return False
    
    vector = np.array(vector)
    norm = np.sqrt(np.sum(vector**2))

    if abs(norm-1) < epsilon:
        return True
    else:
        return False

def check_SEn(matrix, epsilon) -> bool:
    
    m = np.array(matrix)
    n = m.shape[0]

    if n > 4 or n < 3:
        return False
    
    if n == 3:
        rsize = 2
    else:
        rsize = 3

    if m.shape != (n,n):
        return False
    
    rmatrix = m[0:rsize, 0:rsize]
    t = m[0:rsize, -1]
    br = m[-1,-1]

    if abs(br -1) > epsilon:
        return False
    
    if not check_SOn(m[0:rsize,0:rsize],0.01):
        return False

    return True


def correct_SOn(matrix, epsilon):
    m = matrix
    matrix = np.array(matrix)
    u, s, vT = np.linalg.svd(matrix)
    R = np.dot(u, vT)

    if len(m) == 2:
        a, b = R[0]
        c, d = R[1]
        det = a * d - b * c
    else:
        a, b, c = R[0]
        d, e, f = R[1]
        g, h, i = R[2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det - 1) > epsilon:
        u[:, -1] = -u[:, -1] 
        R = np.dot(u, vT)
    
    return R.tolist()

def correct_quaternion(vector, epsilon=0.01):
    if len(vector) != 4:
        print('Quaternion must be vector with 4 components')
        return
    vector = np.array(vector)
    norm = np.sqrt(np.sum(vector**2))
    
    if abs(norm - 1) > epsilon:
        vector = vector / norm  
    
    return vector.tolist()

def correct_SEn(matrix, epsilon=0.01):
    m = np.array(matrix)
    n = m.shape[0]

    if n == 3:
        rsize = 2
    elif n == 4:
        rsize = 3
    else:
        print("Matrix is not in SE(2) or SE(3)")
        return

    rotation_part = m[:rsize, :rsize]
    corrected_rotation = correct_SOn(rotation_part, epsilon)
    
    corrected_matrix = np.eye(n)
    corrected_matrix[:rsize, :rsize] = corrected_rotation

    corrected_matrix[:rsize, -1] = m[:rsize, -1]

    corrected_matrix[-1, -1] = 1
    
    return corrected_matrix.tolist()


def main():
    # Test case for check_SOn (2x2 rotation matrix)
    print("Testing check_SOn (SO(2)):")
    so2_matrix = [[0, -1], [1, 0]]  # 90-degree rotation matrix
    print(f"SO(2) Test Matrix: {so2_matrix}, Result: {check_SOn(so2_matrix,0.01)}")  # Expected: True
    
    # Test case for check_quaternion
    print("\nTesting check_quaternion:")
    quaternion = [0, 0, 0, 1]  # Valid unit quaternion
    print(f"Quaternion: {quaternion}, Result: {check_quaternion(quaternion,0.01)}")  # Expected: True

    # Test case for correct_quaternion
    print("\nTesting correct_quaternion:")
    non_unit_quaternion = [0, 0, 0, 1.5]  # Non-unit quaternion
    corrected_quaternion = correct_quaternion(non_unit_quaternion)
    print(f"Original Quaternion: {non_unit_quaternion}, Corrected Quaternion: {corrected_quaternion}")  # Expected: [0, 0, 0, 1]

    # Test case for check_SEn (SE(3) matrix)
    print("\nTesting check_SEn (SE(3)):")
    se3_matrix = [[0, -1, 0, 1],  # 90-degree rotation about the z-axis with translation
                  [1, 0, 0, 2],
                  [0, 0, 1, 3],
                  [0, 0, 0, 1]]
    print(f"SE(3) Test Matrix: {se3_matrix}, Result: {check_SEn(se3_matrix,0.01)}")  # Expected: True

    # Test case for correct_SEn
    print("\nTesting correct_SEn (SE(3)):")
    faulty_se3_matrix = [[0, -1, 0, 1],  # Slightly incorrect rotation and translation
                         [1, 0, 0, 2],
                         [0, 0, 1, 3],
                         [0, 0, 0, 0.99]]
    corrected_se3 = correct_SEn(faulty_se3_matrix)
    print(f"Original SE(3) Matrix: {faulty_se3_matrix}, Corrected SE(3) Matrix: {corrected_se3}")


if __name__ == "__main__":
    main()
