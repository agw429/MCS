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
        # print('i_sums: {} | j_sums: {}'.format(i_sums, j_sums))
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

        # Calculate marginal probabilities
        n = sum(confusion_mat.values())
        p_i = [i / n for i in i_sums]
        p_j = [j / n for j in j_sums]

        # Calculate mutual information
        mi = 0
        for key in confusion_mat.keys():
            i, j = key
            pij = confusion_mat[key] / n
            if pij > 0:
                mi += pij * math.log(pij / (p_i[i] * p_j[j]))

        # Calculate entropy
        h_i = -sum(pi * math.log(pi) if pi > 0 else 0 for pi in p_i)
        h_j = -sum(pj * math.log(pj) if pj > 0 else 0 for pj in p_j)

        # Calculate NMI
        nmi = mi / (math.sqrt(h_i)*h_j)
        return nmi
