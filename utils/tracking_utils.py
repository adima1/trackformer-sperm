import torch
import torch.nn.functional as F

def match_track_embeddings(embeddings_t, embeddings_tp1, threshold=0.7):
    """
    משווה בין track embeddings של פריים t ופריים t+1.

    :param embeddings_t: Tensor of shape [N1, D] – מהפריים הקודם
    :param embeddings_tp1: Tensor of shape [N2, D] – מהפריים הנוכחי
    :param threshold: סף מינימלי לדמיון
    :return: dict – התאמות: {index_tp1: index_t}
    """
    # ננרמל ליחידות
    emb_t = F.normalize(embeddings_t, dim=1)     # [N1, D]
    emb_tp1 = F.normalize(embeddings_tp1, dim=1) # [N2, D]

    # מחשבים מטריצת דמיון קוסינוס [N2, N1]
    similarity = torch.matmul(emb_tp1, emb_t.T)

    # מוצאים לכל query בפריים t+1 את ההתאמה הכי טובה מהעבר
    matched_indices = {}
    for i in range(similarity.shape[0]):
        max_sim, j = torch.max(similarity[i], dim=0)
        if max_sim.item() > threshold:
            matched_indices[i] = j.item()  # query i עכשיו ← query j מהעבר

    return matched_indices
