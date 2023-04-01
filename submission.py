from typing import Dict, List, Tuple
import math

class Solution:
    def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
        """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
        Args:
          true_labels: list of true labels
          pred_labels: list of predicted labels
        Returns:
          A dictionary of (true_label, pred_label): count
        """
        # matrix = {}
        # matrix[(1, 1)] = matrix[(1, 0)] = matrix[(0, 1)] = matrix[(0, 0)] = 0
        # for true, pred in zip(true_labels, pred_labels):
        #     matrix[(true, pred)] += 1

        matrix = {}
        for true, pred in zip(true_labels, pred_labels):
            if (true, pred) not in matrix:
                matrix[(true, pred)] = 1
            else:
                matrix[(true, pred)] += 1

        sorted_keys = sorted(matrix.keys())
        sorted_data = {k: matrix[k] for k in sorted_keys}
        return sorted_data


    def confusion_mat_sums(self, confusion_mat) -> Tuple[List[int], List[int]]:
        max_i, max_j = max(confusion_mat.keys(), key=lambda x: (x[0], x[1]))
        i_sums = [0] * (max_i + 1)
        j_sums = [0] * (max_j + 1)
        
        for key in confusion_mat.keys():
            i, j = key 
            val = confusion_mat[key]
            i_sums[j] += val
            j_sums[i] += val
        print('i_sums: {} | j_sums: {}'.format(i_sums, j_sums))
        return i_sums, j_sums

    def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the Jaccard index.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The Jaccard index. Do NOT round this value.
        """
        confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        i_sums, j_sums = self.confusion_mat_sums(confusion_mat)

        # m = sum(i_sums)

        tp = fp = fn = tn = 0
        tp = sum(nij*(nij-1)/2 for nij in confusion_mat.values() if nij > 0)
        fp = sum(ni*(ni-1)/2 for ni in i_sums if ni > 0) - tp
        fn = sum(nj*(nj-1)/2 for nj in j_sums if nj > 0) - tp
        # tn = m*(m-1)/2 - tp - fn - fp

        # print("j = {} / ({} + {} + {})".format(tp,tp,fn,fp))
        return (tp / (tp + fn + fp))

    
    def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the normalized mutual information.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The normalized mutual information. Do NOT round this value.
        """
        confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        i_sums, j_sums = self.confusion_mat_sums(confusion_mat)
        m = sum(i_sums)

        print(confusion_mat)
        expected_mat = confusion_mat.copy()
        for i in range(len(i_sums)):
            for j in range(len(j_sums)):
                expected_mat[(i,j)] = i_sums[i] * j_sums[j] / m

        Hi = -sum((val/m)*math.log(val/m) for val in i_sums)
        Hj = -sum((val/m)*math.log(val/m) for val in j_sums)
        Iij = 0
        for key in confusion_mat:
            val = confusion_mat[key]
            exp = expected_mat[key]
            print("key: {} | val: {} | exp: {}".format(key, val, exp))
            if val > 0: 
                tmp = (val/m)*math.log((val/m) / (exp / m)) 
                Iij += tmp
                print("Iij += ({}/{})*math.log(({}/{})) += {}".format(val, m, val, exp, tmp))

        print("NMI = {} / (sqrt({})({}))".format(round(Iij,4), round(Hi,4), round(Hj,4)))
        return Iij / (math.sqrt(Hi) * Hj)