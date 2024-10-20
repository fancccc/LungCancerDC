import torch
from torch import nn
from src.resnet import generate_model

class QKV(nn.Module):
    def __init__(self):
        super(QKV, self).__init__()
        self.q_proj = nn.Linear(in_features=512, out_features=512)
        self.k_proj = nn.Linear(in_features=512, out_features=512)
        self.v_proj = nn.Linear(in_features=512, out_features=512)

class ModiResnet(nn.Module):
    def __init__(self, rd: int=18,
                 ):
        super().__init__()
        self.image_emcoder = generate_model(model_depth=rd, n_classes=3)
        # model = torch.load('/home/zcd/codes/LungCancerDC/results/2408031253/models/model_best_test.pt') # resnet18 ct32
        # model = torch.load('/home/zcd/codes/LungCancerDC/results/2408040923/models/model_best_test.pt')  # resnet18 ct128

        # model_state_dict = model['model_state_dict']
        # self.image_emcoder.load_state_dict(model_state_dict)
        self.bbox_proj = nn.Linear(in_features=6, out_features=self.image_emcoder.fc.in_features)
        self.image_emcoder = nn.Sequential(*list(self.image_emcoder.children())[:-1])

        self.att = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)


    def forward(self, data):
        img, bbox = data['image'], data['bbox']
        # x = (img, cli)
        # x = self.enhance_image(x) # img, box = x
        x1 = self.image_emcoder(img) # b, 512, 1, 1, 1
        x1 = x1.flatten(2).transpose(1, 2) # b, 1, 512
        x2 = self.bbox_proj(bbox) # b, 512
        att_out, weights = self.att(x2.unsqueeze(1), x1, x1)
        # x = torch.cat([x1, x2.unsqueeze(1)], dim=1)

        # print(att_out.shape, weights)
        return att_out


if __name__ == '__main__':
    from src.dataloader import split_pandas, LungDataset, DataLoader
    train_info, val_info = split_pandas('../configs/dataset.json')
    train_dataset = LungDataset(train_info, '../configs/dataset.json', use_ct32=True, use_ct128=True, use_radiomics=True, use_cli=True, use_bbox=True)
    val_dataset = LungDataset(val_info, '../configs/dataset.json', phase='val', use_ct32=True, use_ct128=True, use_radiomics=True, use_cli=True, use_bbox=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=2,
                                shuffle=True,
                                num_workers=6
                                    )
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=6
                                    )
    model = ModiResnet(rd=18)
    data = next(iter(train_loader))
    data['image'] = data['ct128']
    data['bbox'] = data['bbox32']
    model(data)
