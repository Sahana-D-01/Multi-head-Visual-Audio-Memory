import torch
from torch import nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, radius=16.0, n_slot=112, n_head=8, dim=512, diff_aud_vid=False):
        #number of slots in one key memory
        #number of heads in one key memory
        #dim is the embedding dimension of each memory slot; number of features used to represent each slot
        super().__init__()

        self.dav = diff_aud_vid
        self.head = n_head
        self.slot = n_slot
        

        self.key = nn.Parameter(torch.Tensor(int(n_head * n_slot), int(512 / n_head)), requires_grad=True)
        #defining the dimensions of the memory network
        #each head is allocated a dimension of 64; there are 8 heads: 64*8
        
        nn.init.normal_(self.key, 0, 0.5)
        self.value = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        #checking if it trained on different modalities of input data
        self.linear = nn.Linear(512 * n_head, dim)
        #applying linear transformation to reduce dimensionality to "dim" while maintaining important information
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.v_up = nn.Linear(512, dim)
        #projects concatenated features onto lower dimensionality

        self.q_embd = nn.Linear(dim, 512)
        #embed incoming vectors into space of 512
        self.v_embd = nn.Linear(512, 512)
        #embed stored value features into space of 512
        
        self.dropout = nn.Dropout(0.5)

        self.radius = radius
        self.softmax1 = nn.Softmax(2)
        self.softmax2 = nn.Softmax(1)

    def forward(self, query, value=None, inference=False, cha_first=False):
        if cha_first:
            query = query.transpose(1, 2).contiguous()
        # B, S, 512
        B, S, C = query.size()
        #Batch size, sequence length, number of channels
        mer_query = query.view(B * S, -1)
        tr_fusion, recon_loss, add_loss = None, torch.zeros(1).cuda(), torch.zeros(1).cuda()

        key_norm = F.normalize(self.key.view(self.head, self.slot, -1), dim=2) #n_head, n_slot, head_dim
        embd_query = self.q_embd(mer_query) #B*S, n_head * head_dim
        #reshape query into (B*S, -1) dimensions
        embd_query = embd_query.view(B * S, self.head, -1)  #BS, n_head, head_dim
        embd_query = F.normalize(embd_query, dim=2)

        key_sim = torch.einsum('bhd,hsd->bhs', embd_query, key_norm)    #BS, n_head, n_slot
        #cosine similarity calculation of incoming frames
        key_add = self.softmax1(self.radius * key_sim)  # BS, n_head, n_slot
        #addressing vector calculation of key
        
        m_head_aud = torch.matmul(key_add, self.value.detach()) # BS, n_head, 512
        #fetching audio wrt value feature

        te_fusion = self.norm1(query + vir_aud.view(B, S, -1))
        te_fusion = self.dropout(te_fusion)

        # Update
        if not inference:
            mer_value = value.view(B * S, -1)   #BS,512
            embd_value = self.v_embd(mer_value.detach())
            value_norm = F.normalize(self.value, dim=1) #n_slot,512
            value_sim = F.linear(F.normalize(embd_value, dim=1), value_norm) #BS, n_slot
            #cosine similarity calculation of audio
            value_add = self.softmax2(self.radius * value_sim)
            #addressing vector calculation of value
            
            aud = torch.matmul(value_add, self.value)   #BS,512
            #audio renconstruction wrt key feature
            
            add_loss = F.kl_div(torch.log(key_add), value_add.detach(), reduction='batchmean')
            add_loss = add_loss.unsqueeze(0)
            #kullback-divergence loss calculation
            
            recon_loss = torch.abs(1.0 - F.cosine_similarity(aud, mer_value.detach(), 1))  #BS
            recon_loss = recon_loss.view(B, S).sum(1)   #B
            #reconstructive loss calculation
            
            if self.dav:
                aud = self.v_up(aud)
            aud = self.norm3(aud)
            tr_fusion = self.norm1(query + aud.view(B, S, -1))
            tr_fusion = self.dropout(tr_fusion)
            #concatenated features to be fed into the decoder
        if cha_first:
            te_fusion = te_fusion.transpose(1, 2).contiguous()
            if tr_fusion is not None:
                tr_fusion = tr_fusion.transpose(1, 2).contiguous()
        return te_fusion, tr_fusion, recon_loss, add_loss
