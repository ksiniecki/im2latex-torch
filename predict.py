import argparse
import torch

from os.path import join
from model import Im2LatexModel, LatexProducer
from build_vocab import Vocab, load_vocab

from PIL import Image
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description="Im2Latex Predict")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument('--model_path', required=True,
                        help='path of the evaluated model')
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    parser.add_argument("--im_path", type=str,
                        required=True, help="The path to image")
    parser.add_argument("--dropout", type=float,
                        default=0., help="Dropout probility")
    parser.add_argument("--max_len", type=int,
                        default=150, help="Max size of formula")
    parser.add_argument("--add_position_features", action='store_true',
                        default=False, help="Use position embeddings or not")
    parser.add_argument("--emb_dim", type=int,
                        default=80, help="Embedding size")
    parser.add_argument("--dec_rnn_h", type=int, default=512,
                        help="The hidden state of the decoder RNN")
    parser.add_argument("--seed", type=int, default=2020,
                        help="The random seed for reproducing ")
    parser.add_argument("--beam_size", type=int, default=5)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Load vocab...")
    vocab = load_vocab(args.data_path)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint = torch.load(join(args.model_path), map_location=torch.device(device))
    model_args = checkpoint['args']

    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size)
    
    transform = transforms.ToTensor()
    img = Image.open(args.im_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    result = latex_producer(img_tensor)
    print('Prediction:', result[0])

if __name__ == "__main__":
    main()
