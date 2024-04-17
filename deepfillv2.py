from PIL import Image
import torch
import torchvision.transforms as T


def inpaint(image, mask, model='states_tf_celebahq'):
    generator_state_dict = torch.load(f'pretrained/{model}.pth')['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                                    and use_cuda_if_available else 'cpu')

    # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(f'pretrained/{model}.pth')['G']
    generator.load_state_dict(generator_state_dict, strict=True)

    # prepare input
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h // grid * grid, :w // grid * grid].unsqueeze(0)
    mask = mask[0:1, :h // grid * grid, :w // grid * grid].unsqueeze(0)

    image = (image * 2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1. - mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x * mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1. - mask) + x_stage2 * mask

    # save inpainted image
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    return img_out
