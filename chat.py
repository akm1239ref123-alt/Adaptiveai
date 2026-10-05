import argparse, torch
from persona.model import PersonaGPT
from persona.tokenizer import ByteTokenizer

a=argparse.ArgumentParser(); a.add_argument('--prompt',default='User: Hello\nPersona:'); a.add_argument('--steps',type=int,default=200); a.add_argument('--temperature',type=float,default=.7); args=a.parse_args()
ck=torch.load('checkpoints/persona-20m.pt',map_location='cpu',weights_only=False)
tok=ByteTokenizer.load('checkpoints/tokenizer-20m.json')
model=PersonaGPT(**ck['config']); model.load_state_dict(ck['model']); model.eval()
ids=tok.encode(args.prompt); x=torch.tensor([ids],dtype=torch.long)
out=model.generate(x,args.steps,args.temperature,50,.95)[0].tolist()
print(tok.decode(out))
