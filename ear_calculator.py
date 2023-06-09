def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2])**2).sum())**0.5

def calculate_ear(mesh, res):
    right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
    left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions

    N1 = distance(mesh[right_eye[1][0]], mesh[right_eye[1][1]])
    N2 = distance(mesh[right_eye[2][0]], mesh[right_eye[2][1]])
    N3 = distance(mesh[right_eye[3][0]], mesh[right_eye[3][1]])
    D = distance(mesh[right_eye[0][0]], mesh[right_eye[0][1]])
    res.append((N1 + N2 + N3) / (3 * D))

    N1 = distance(mesh[left_eye[1][0]], mesh[left_eye[1][1]])
    N2 = distance(mesh[left_eye[2][0]], mesh[left_eye[2][1]])
    N3 = distance(mesh[left_eye[3][0]], mesh[left_eye[3][1]])
    D = distance(mesh[left_eye[0][0]], mesh[left_eye[0][1]])
    res.append((N1 + N2 + N3) / (3 * D))


def calculate_mar(mesh, res):
    mouth = [[78, 308], [13, 14], [81, 178], [311, 402]]

    N1 = distance(mesh[mouth[1][0]], mesh[mouth[1][1]])
    N2 = distance(mesh[mouth[2][0]], mesh[mouth[2][1]])
    N3 = distance(mesh[mouth[3][0]], mesh[mouth[3][1]])
    D = distance(mesh[mouth[0][0]], mesh[mouth[0][1]])
    res.append((N1 + N2 + N3) / (3 * D))