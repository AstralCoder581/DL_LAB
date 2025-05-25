import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

def load_img(path, device):
    img = Image.open(path).convert('RGB')
    tfm = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return tfm(img).unsqueeze(0).to(device)

def gram(x):
    b, c, h, w = x.size()
    f = x.view(c, h*w)
    return torch.mm(f, f.t())

def get_feats(x, model, layers):
    feats, curr = {}, x
    for i, layer in model._modules.items():
        curr = layer(curr)
        if i in layers: feats[layers[i]] = curr
    return feats

def style_transfer(content_path, style_path, steps=300, device='cpu'):
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
    for p in vgg.parameters(): p.requires_grad = False
    layers = {'0':'c1', '5':'c2', '10':'c3', '19':'c4', '21':'c5'}
    style_w = {'c1':1, 'c2':0.8, 'c3':0.5, 'c4':0.3}
    content, style = load_img(content_path, device), load_img(style_path, device)
    target = content.clone().requires_grad_(True)
    c_feats = get_feats(content, vgg, layers)
    s_grams = {l: gram(f) for l, f in get_feats(style, vgg, layers).items()}
    opt = torch.optim.Adam([target], lr=0.003)
    for i in range(steps):
        t_feats = get_feats(target, vgg, layers)
        c_loss = torch.mean((t_feats['c5']-c_feats['c5'])**2)
        s_loss = sum(style_w[l]*torch.mean((gram(t_feats[l])-s_grams[l])**2) for l in style_w)
        loss = c_loss + 1e6*s_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if i%50==0: print(f'Step {i}: {loss.item():.2f}')
    return target

# Example usage:
# result = style_transfer('content.jpg', 'style.jpg', device='cpu')
result = style_transfer('/kaggle/input/cars-dataset/car1.jpeg', '/kaggle/input/cars-dataset/car2.jpg')
