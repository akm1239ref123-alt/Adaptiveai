import argparse, torch
from persona.model import PersonaGPT
from persona.tokenizer import CharTokenizer

a=argparse.ArgumentParser(); a.add_argument('--prompt',default='Persona:'); a.add_argument('--steps',type=int,default=200); a.add_argument('--temperature',type=float,default=.7); args=a.parse_args()
ck=torch.load('checkpoints/persona-v0.pt',map_location='cpu',weights_only=False)
tok=CharTokenizer.load('checkpoints/tokenizer.json')
model=PersonaGPT(**ck['config']); model.load_state_dict(ck['model']); model.eval()
ids=tok.encode(args.prompt)
x=torch.tensor([ids],dtype=torch.long)
out=model.generate(x,args.steps,args.temperature,40)[0].tolist()
print(tok.decode(out))
