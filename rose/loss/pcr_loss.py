import torch
import torch.nn as nn
import torch.nn.functional as F

class PCRLoss(nn.Module):
    def __init__(
            self,
            num_global_crops=2,
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.num_global_crops = num_global_crops

    def forward(self, student_feats, teacher_feats, pcr_bin_matrix):
        """
        Compute the PCR loss between the student and teacher features.
        """

        student_feats = F.normalize(student_feats, dim=-1)  # [B, N, D]
        teacher_feats = F.normalize(teacher_feats, dim=-1)  # [B, N, D]

        # teacher_feats_reorder = torch.cat([teacher_feats[B:], teacher_feats[:B]])
        pos_mask = pcr_bin_matrix.bool()  # [B, N, D]
        sim_matrix_s2t = torch.matmul(student_feats, teacher_feats.transpose(-1, -2))
        # 1. scale logits
        logits = sim_matrix_s2t / 0.1   # [B, N, N]

        # 2. get all positive index triplets (b,i,j_pos)
        pos_indices = pos_mask.nonzero(as_tuple=False)   # [P, 3]

        if pos_indices.numel() == 0:
            return torch.tensor(0., device=logits.device)

        b_idx = pos_indices[:, 0]
        i_idx = pos_indices[:, 1]
        j_pos = pos_indices[:, 2]

        P = len(j_pos)
        _, N, _ = logits.shape

        # 3. for each positive, construct one CE training sample
        # logits_expanded[k] = logits[b_k, i_k, :]
        logits_expanded = logits[b_idx, i_idx, :]     # [P, N]

        # 4. exclude other positives (only keep current pos + negatives)
        # other_pos_mask[k] = pos_mask[b_k, i_k, :] but excluding j_pos[k]
        all_pos_mask = pos_mask[b_idx, i_idx, :]      # [P, N]
        # remove current positive (set it to False temporarily)
        mask_without_self = all_pos_mask.clone()
        mask_without_self[torch.arange(P), j_pos] = False

        # set other positives to -inf
        logits_expanded = logits_expanded.masked_fill(mask_without_self, float('-inf'))

        # target for CE is the index of the current pos
        target_expanded = j_pos    # [P]

        # 5. compute CE loss
        loss = F.cross_entropy(logits_expanded, target_expanded)
        # negate_bin_matrix = -bi_bin_matrix + 1
        
        # positive_sim = bi_bin_matrix * (sim_matrix_s2t)
        # negative_sim = negate_bin_matrix * (sim_matrix_s2t)
        # pcr_loss = -positive_sim.mean() + negative_sim.mean()
        '''
        # float casting
        student_feats = student_feats.flatten(0,1)
        teacher_feats = teacher_feats.flatten(0,1)

        student_feats = F.normalize(student_feats, dim=-1)  # [2*B, N, D]
        teacher_feats = F.normalize(teacher_feats, dim=-1)  # [2*B, N, D]
        
        # bi here represents 2 directions v->u & u->v 
        B = teacher_feats.shape[0] // 2
        teacher_feats_reorder = torch.cat([teacher_feats[B:], teacher_feats[:B]])
        sim_matrix_s_t = torch.exp(torch.matmul(student_feats, teacher_feats_reorder.transpose(-1,-2)) / 0.1)
        bi_bin_matrix = torch.cat([pcr_bin_matrix, pcr_bin_matrix.transpose(-1,-2)], dim=0)  # [2*B, N, N]
        bi_positive_set = bi_bin_matrix * sim_matrix_s_t  # [2*B, N, N]

        valid_rows = (bi_positive_set.sum(dim=-1) > 0) & (bi_bin_matrix.sum(dim=-1) < bi_bin_matrix.shape[-1])
        valid_cols = (bi_positive_set.sum(dim=-2) > 0) & (bi_bin_matrix.sum(dim=-2) < bi_bin_matrix.shape[-2])

        row_Infonce = torch.log((bi_positive_set.sum(dim=-1)) / (sim_matrix_s_t.sum(dim=-1) + 1e-6))[valid_rows]
        col_Infonce = torch.log((bi_positive_set.sum(dim=-2)) / (sim_matrix_s_t.sum(dim=-2) + 1e-6))[valid_cols]

        loss_bi_ij = -(row_Infonce.mean() if row_Infonce.numel() > 0 else torch.tensor(0.0))
        loss_bi_ji = -(col_Infonce.mean() if col_Infonce.numel() > 0 else torch.tensor(0.0))

        loss = 0.5 * (loss_bi_ij + loss_bi_ji)
        '''
        return loss