import argparse, json, math, time
from pathlib import Path
import torch
from persona.model import PersonaGPT
from persona.tokenizer import ByteTokenizer

p = argparse.ArgumentParser(description='Train Persona 20M from scratch.')
p.add_argument('--steps', type=int, default=50000)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--grad-accum', type=int, default=8)
p.add_argument('--block-size', type=int, default=512)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--min-lr', type=float, default=2e-5)
p.add_argument('--n-embd', type=int, default=448)
p.add_argument('--n-head', type=int, default=8)
p.add_argument('--n-layer', type=int, default=8)
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--data', default='data/train_20m.txt')
p.add_argument('--out', default='checkpoints/persona-20m.pt')
p.add_argument('--resume', action='store_true')
p.add_argument('--eval-every', type=int, default=250)
p.add_argument('--save-every', type=int, default=1000)
p.add_argument('--seed', type=int, default=1337)
a = p.parse_args()

torch.manual_seed(a.seed)
root = Path(__file__).parent
text_path = root / a.data
feedback = root / 'data' / 'feedback.jsonl'
text = text_path.read_text(encoding='utf-8')
if feedback.exists():
    for line in feedback.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            if int(row.get('rating', 0)) > 0:
                text += f"\nUser: {row.get('prompt','')}\nPersona: {row.get('response','')}\n"
            if row.get('correction'):
                text += f"\nUser: {row.get('prompt','')}\nPersona: {row.get('correction','')}\n"
        except Exception:
            pass

tok = ByteTokenizer()
tok.save(root / 'checkpoints' / 'tokenizer-20m.json')
ids = torch.tensor(tok.encode(text), dtype=torch.long)
cut = max(a.block_size + 2, int(len(ids) * 0.95))
train_ids, val_ids = ids[:cut], ids[cut:]
if len(val_ids) <= a.block_size + 1:
    val_ids = train_ids[-min(len(train_ids), a.block_size * 100):]

def batch(data):
    max_start = len(data) - a.block_size - 1
    ix = torch.randint(max_start + 1, (a.batch_size,))
    x = torch.stack([data[i:i+a.block_size] for i in ix])
    y = torch.stack([data[i+1:i+a.block_size+1] for i in ix])
    return x, y

cfg = dict(vocab_size=256, block_size=a.block_size, n_embd=a.n_embd,
           n_head=a.n_head, n_layer=a.n_layer, dropout=a.dropout)
model = PersonaGPT(**cfg)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))

start_step = 0
if a.resume and Path(a.out).exists():
    ck = torch.load(a.out, map_location='cpu', weights_only=False)
    if ck.get('config') != cfg:
        raise RuntimeError(f'Checkpoint config does not match current config: {ck.get("config")} != {cfg}')
    model.load_state_dict(ck['model'])
    if 'optimizer' in ck:
        opt.load_state_dict(ck['optimizer'])
    start_step = int(ck.get('step', 0))
    print(f'resumed {a.out} at step {start_step}')

params = sum(x.numel() for x in model.parameters())
print(f'Persona 20M: {params:,} parameters; corpus={len(text):,} chars (~{len(ids):,} byte tokens)')
print(f'architecture: {a.n_layer} layers / {a.n_embd} hidden / {a.n_head} heads / {a.block_size} context')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
use_amp = device.type == 'cuda'
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

# Keep data on CPU and transfer only each batch, which makes the script usable on modest machines.
def get_batch(data):
    x, y = batch(data)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def lr_at(step):
    if step <= max(100, a.steps // 100):
        return a.lr * step / max(1, a.steps // 100)
    progress = (step - max(100, a.steps // 100)) / max(1, a.steps - max(100, a.steps // 100))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return a.min_lr + (a.lr - a.min_lr) * cosine

best_val = float('inf')
for step in range(start_step + 1, a.steps + 1):
    model.train()
    opt.zero_grad(set_to_none=True)
    running = 0.0
    for _ in range(a.grad_accum):
        x, y = get_batch(train_ids)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
            loss = loss / a.grad_accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running += loss.item()
    lr = lr_at(step)
    for g in opt.param_groups: g['lr'] = lr
    if scaler.is_enabled():
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler.is_enabled(): scaler.step(opt); scaler.update()
    else: opt.step()

    if step == start_step + 1 or step % a.eval_every == 0:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch(val_ids)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, vl = model(vx, vy)
        val = float(vl.item())
        print(f'step {step:6d} train={running:.4f} val={val:.4f} lr={lr:.2e}')
        if val < best_val:
            best_val = val
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg, 'step': step, 'best_val': best_val}, root / 'checkpoints' / 'persona-20m-best.pt')
    if step % a.save_every == 0:
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg, 'step': step, 'best_val': best_val}, root / a.out)
        print('checkpoint saved', a.out)

torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg, 'step': a.steps, 'best_val': best_val}, root / a.out)
print('saved', a.out)
