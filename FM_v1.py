# =========================================================
# IMPORT
# =========================================================
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from datetime import datetime
import math
import random
import copy

# =========================================================
# ID MAPPER
# =========================================================
class IDMapper:
    def __init__(self):
        self.map = {}

    def get(self, key):
        if key not in self.map:
            self.map[key] = len(self.map)
        return self.map[key]

# =========================================================
# LOAD USER CONTEXT
# =========================================================
def load_user_context(path):
    ctx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)

            # yelping_since → số năm hoạt động
            try:
                dt = datetime.strptime(u.get("yelping_since", "2015-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
                years_active = max(0.0, (2020 - dt.year) / 10.0)
            except:
                years_active = 0.5

            # elite count
            elite_str = u.get("elite", "")
            #elite_count = len(elite_str.split(",")) if elite_str else 0
            
            if isinstance(elite_str, list):
                elite_count = len(elite_str)
            else:
                elite_count = len(elite_str.split(",")) if elite_str else 0

            # friends count
            friends = u.get("friends", "")
            if isinstance(friends, str) and friends and friends != "None":
                friend_count = len([f for f in friends.split(",") if f.strip()])
            else:
                friend_count = 0
            compliment_keys = [
                "compliment_hot", "compliment_more", "compliment_profile",
                "compliment_cute", "compliment_list", "compliment_note",
                "compliment_plain", "compliment_cool", "compliment_funny",
                "compliment_writer", "compliment_photos"
            ]
            total_compliments = sum(u.get(k, 0) or 0 for k in compliment_keys)
            ctx[u["user_id"]] = [
                math.log1p(u.get("review_count", 0)) / 10.0,
                u.get("average_stars", 0) / 5.0,
                math.log1p(u.get("fans", 0)) / 10.0,
                math.log1p(max(0, u.get("useful", 0) or 0)) / 10.0,
                math.log1p(max(0, u.get("funny",  0) or 0)) / 10.0,
                math.log1p(max(0, u.get("cool",   0) or 0)) / 10.0,
                years_active,
                math.log1p(elite_count) / 10.0, 
                math.log1p(friend_count) / 10.0,
                math.log1p(total_compliments) / 10.0,
            ]
    return ctx

# =========================================================
# LOAD BUSINESS CONTEXT
# =========================================================
def load_business_context(path):
    ctx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)

            # categories count
            categories = b.get("categories", "")
            cat_count = len(categories.split(",")) if categories else 0

            # hours count (how many days open)
            hours = b.get("hours", {})
            open_days = len(hours) if hours else 0

            ctx[b["business_id"]] = [
                b.get("stars", 0) / 5.0,
                math.log1p(b.get("review_count", 0)) / 10.0,
                float(b.get("is_open", 0)),
                b.get("latitude", 0) / 90.0,
                b.get("longitude", 0) / 180.0,
                math.log1p(cat_count) / 10.0,
                open_days / 7.0,
            ]
    return ctx

# =========================================================
# LOAD CHECKIN CONTEXT
# =========================================================
def load_checkin_context(path):
    ctx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            bid = c["business_id"]
            dates = c.get("date", "")

            if dates:
                times = dates.split(", ")
                count = len(times)

                # density theo tháng (rough)
                density = count / 30.0
            else:
                count = 0
                density = 0

            ctx[bid] = [
                math.log1p(count) / 10.0,
                density
            ]
    return ctx

# =========================================================
# LOAD TIP CONTEXT
# =========================================================
def load_tip_context(path):
    user_tip = {}
    biz_tip = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            uid = t["user_id"]
            bid = t["business_id"]
            length = len(t.get("text", ""))
            comp = t.get("compliment_count", 0)

            if uid not in user_tip:
                user_tip[uid] = [0, 0, 0]
            user_tip[uid][0] += 1
            user_tip[uid][1] += length
            user_tip[uid][2] += comp

            if bid not in biz_tip:
                biz_tip[bid] = [0, 0, 0]
            biz_tip[bid][0] += 1
            biz_tip[bid][1] += length
            biz_tip[bid][2] += comp

    return user_tip, biz_tip

# =========================================================
# STREAMING DATASET
# =========================================================

USER_CTX_DIM = 10
BIZ_CTX_DIM  = 7
TIP_DIM      = 3
CHECKIN_DIM  = 2
REVIEW_DIM   = 4
TEMPORAL_DIM = 3

NUM_DIM = USER_CTX_DIM + BIZ_CTX_DIM + TIP_DIM * 2 + CHECKIN_DIM + REVIEW_DIM + TEMPORAL_DIM  # = 32
class YelpDataset(IterableDataset):
    def __init__(
        self,
        review_path,
        user_ctx,
        biz_ctx,
        user_tip,
        biz_tip,
        checkin_ctx,
        max_reviews=None,
    ):
        self.review_path = review_path
        self.user_ctx = user_ctx
        self.biz_ctx = biz_ctx
        self.user_tip = user_tip
        self.biz_tip = biz_tip
        self.checkin_ctx = checkin_ctx
        self.max_reviews = max_reviews
        self.user_map = IDMapper()
        self.biz_map = IDMapper()

    def __iter__(self):
        count = 0
        with open(self.review_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_reviews and count >= self.max_reviews:
                    break

                r = json.loads(line)
                uid = r["user_id"]
                bid = r["business_id"]

                uidx = self.user_map.get(uid)
                bidx = self.biz_map.get(bid)

                # USER
                #uctx = self.user_ctx.get(uid, [0]*10)
                uctx = self.user_ctx.get(uid, [0] * USER_CTX_DIM)
                
                # BUSINESS
                #bctx = self.biz_ctx.get(bid, [0]*7)
                bctx = self.biz_ctx.get(bid, [0] * BIZ_CTX_DIM)
                
                # TIP
                utip = self.user_tip.get(uid, [0, 0, 0])
                btip = self.biz_tip.get(bid, [0, 0, 0])

                utip_feat = [
                    utip[0],
                    utip[1] / max(utip[0], 1),
                    utip[2] / max(utip[0], 1),
                ]
                btip_feat = [
                    btip[0],
                    btip[1] / max(btip[0], 1),
                    btip[2] / max(btip[0], 1),
                ]

                # CHECKIN
                checkin = self.checkin_ctx.get(bid, [0, 0])

                # REVIEW
                useful   = max(0, r.get("useful", 0) or 0)
                funny    = max(0, r.get("funny",  0) or 0)
                cool     = max(0, r.get("cool",   0) or 0)
                text_len = len(r.get("text", "") or "")
                dt = datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S")

                # Normalize temporal features
                year_norm = (dt.year - 2015) / 10.0  # Normalize year
                month_norm = dt.month / 12.0
                weekday_norm = dt.weekday() / 6.0

                # Normalize text length (log scale to handle large values)
                text_len_norm = math.log1p(text_len) / 10.0

                # Normalize counts (log scale)
                useful_norm = math.log1p(useful) / 10.0
                funny_norm  = math.log1p(funny)  / 10.0
                cool_norm   = math.log1p(cool)   / 10.0
                
                # Normalize tip features
                utip_feat_norm = [
                    math.log1p(utip_feat[0]) / 10.0,
                    math.log1p(utip_feat[1]) / 10.0,
                    math.log1p(utip_feat[2]) / 10.0,
                ]
                btip_feat_norm = [
                    math.log1p(btip_feat[0]) / 10.0,
                    math.log1p(btip_feat[1]) / 10.0,
                    math.log1p(btip_feat[2]) / 10.0,
                ]

                # Normalize checkin features
                #checkin_norm = math.log1p(checkin[0]) / 10.0
                checkin_feat = [
                    checkin[0],                          # đã log1p/10 sẵn ✅
                    math.log1p(checkin[1]) / 10.0,      # normalize density ✅
                ]

                num_feat = torch.tensor(
                    uctx
                    + bctx
                    + utip_feat_norm
                    + btip_feat_norm
                    + checkin_feat
                    + [useful_norm, funny_norm, cool_norm, text_len_norm]
                    + [year_norm, month_norm, weekday_norm],
                    dtype=torch.float32
                )

                rating = torch.tensor(r["stars"], dtype=torch.float32)

                yield (
                    torch.tensor(uidx),
                    torch.tensor(bidx),
                    num_feat,
                    rating,
                )
                count += 1

# =========================================================
# FM MODEL
# =========================================================

class FMModel(nn.Module):
    def __init__(self, n_user, n_business, num_dim, k=128, dropout=0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.k = k
        self.num_dim = num_dim
        self.num_bn = nn.BatchNorm1d(num_dim)
        # Linear terms
        self.bias = nn.Parameter(torch.tensor([3.7840]))
        self.user_lin = nn.Embedding(n_user, 1)
        self.biz_lin  = nn.Embedding(n_business, 1)
        self.num_lin  = nn.Parameter(torch.zeros(num_dim))

        # Interaction embeddings
        self.user_emb = nn.Embedding(n_user, k)
        self.biz_emb  = nn.Embedding(n_business, k)
        self.num_emb  = nn.Parameter(torch.zeros(num_dim, k))

        # Init tất cả về std nhỏ
        nn.init.normal_(self.user_lin.weight, 0, 0.01)
        nn.init.normal_(self.biz_lin.weight,  0, 0.01)
        nn.init.normal_(self.num_lin,         0, 0.01)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.biz_emb.weight,  0, 0.01)
        nn.init.normal_(self.num_emb,         0, 0.01)

    def forward(self, user, biz, num):
        num = self.num_bn(num)
        # Linear part
        linear = (
            self.bias
            + self.user_lin(user).squeeze(-1)   # (B,)
            + self.biz_lin(biz).squeeze(-1)      # (B,)
            + (num * self.num_lin).sum(dim=1)    # (B,)
        )

        # Interaction part
        u = self.user_emb(user).unsqueeze(1)     # (B, 1, k)
        b = self.biz_emb(biz).unsqueeze(1)       # (B, 1, k)
        n = num.unsqueeze(-1) * self.num_emb     # (B, F, k)

        all_emb = torch.cat([u, b, n], dim=1)    # (B, F+2, k)
        all_emb = self.dropout(all_emb)
        sum_sq  = all_emb.sum(dim=1) ** 2        # (B, k)
        sq_sum  = (all_emb ** 2).sum(dim=1)      # (B, k)
        inter   = 0.5 * (sum_sq - sq_sum).sum(dim=1)  # (B,)

        out = linear + inter

        if not self.training:
            out = torch.clamp(out, min=1.0, max=5.0)

        return out
        

# =========================================================
# EVALUATION METRICS
# =========================================================
def compute_metrics(predictions, targets):
    
    predictions = predictions.cpu()
    targets = targets.cpu()
    
    # MSE - Mean Squared Error
    mse = torch.mean((predictions - targets) ** 2).item()
    
    # RMSE - Root Mean Squared Error
    rmse = math.sqrt(mse)
    
    # MAE - Mean Absolute Error
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # R2 - R-squared (Coefficient of Determination)
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def evaluate_model(model, loader, device):
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for user, biz, num, rating in loader:
            user = user.to(device)
            biz = biz.to(device)
            num = num.to(device)
            
            pred = model(user, biz, num)
            all_preds.append(pred.cpu())
            all_targets.append(rating)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    model.train()
    
    return metrics

def print_metrics(metrics, prefix=""):
    
    print(f"{prefix}MSE: {metrics['MSE']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f}")

# =========================================================
# TRAIN
# =========================================================
def train_model(
    dataset,
    epochs=120,
    batch_size=2048,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    early_stopping_patience=30,
    early_stopping_min_delta=1e-4,
    eval_every=1,
):
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not math.isclose(train_split + val_split + test_split, 1.0, rel_tol=1e-6):
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    # Collect all data for train/val/test split
    print("Loading data into memory for train/validation/test split...")
    all_data = list(dataset)
    
    # Shuffle and split: 80% train, 10% validation, 10% test (configurable)
    random.shuffle(all_data)
    total_size = len(all_data)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)

    train_end = train_size
    val_end = train_end + val_size

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    print(
        f"Train size: {len(train_data)}, "
        f"Validation size: {len(val_data)}, "
        f"Test size: {len(test_data)}"
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Embedding size lớn để tránh overflow
    # model = FMModel(
    #     n_user=1_000_000,
    #     n_business=200_000,
    #     num_dim=32
    # ).to(device)

    n_user     = len(dataset.user_map.map) + 1000
    n_business = len(dataset.biz_map.map) + 1000
    print(f"Unique users: {n_user - 1000}, Unique businesses: {n_business - 1000}")
    
    model = FMModel(
        n_user=n_user,
        n_business=n_business,
        num_dim=NUM_DIM
    ).to(device)
    
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    # SGD with lower learning rate for stability (features are normalized)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5, nesterov=True)
    loss_fn = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=8,
        factor=0.5,
        min_lr=1e-5,
    )

    print(f"Starting training with learning rate (lr): {optimizer.param_groups[0]['lr']}")
    
    # History for tracking metrics
    history = {
        'train_loss': [],
        'val_mse': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': []
    }

    best_val_rmse = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improve = 0
    
    for ep in range(epochs):

        model.train()
        total_loss = 0
        total_count = 0
        
        for user, biz, num, rating in train_loader:
            user = user.to(device)
            biz = biz.to(device)
            num = num.to(device)
            rating = rating.to(device)
            
            # Check for NaN in input
            if torch.isnan(num).any():
                print("Warning: NaN in input features, skipping batch")
                continue
            
            pred = model(user, biz, num)
            loss = loss_fn(pred, rating)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * len(user)
            total_count += len(user)
        
        avg_loss = total_loss / total_count if total_count > 0 else float('nan')
        history['train_loss'].append(avg_loss)
        
        # Evaluate on validation set
        if (ep + 1) % eval_every == 0:
            val_metrics = evaluate_model(model, val_loader, device)
            scheduler.step(val_metrics['RMSE'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  >> Current LR: {current_lr:.6f}")
            history['val_mse'].append(val_metrics['MSE'])
            history['val_rmse'].append(val_metrics['RMSE'])
            history['val_mae'].append(val_metrics['MAE'])
            history['val_r2'].append(val_metrics['R2'])
            
            print(f"Epoch {ep+1}/{epochs} | Train Loss: {avg_loss:.4f}")
            print_metrics(val_metrics, prefix="  Val -> ")
            
            # Save best model
            if val_metrics['RMSE'] < (best_val_rmse - early_stopping_min_delta):
                best_val_rmse = val_metrics['RMSE']
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = ep + 1
                epochs_without_improve = 0
                print(f"  >> New best model (RMSE: {best_val_rmse:.4f})")
            else:  
                epochs_without_improve += 1
                print(
                    f"  >> No improvement for {epochs_without_improve}/"
                    f"{early_stopping_patience} eval(s)"
                )
                if epochs_without_improve >= early_stopping_patience:
                    print(
                        f"\nEarly stopping at epoch {ep+1}: "
                        f"Val RMSE did not improve by at least {early_stopping_min_delta} "
                        f"for {early_stopping_patience} consecutive eval(s)."
                    )
                    history['stopped_epoch'] = ep + 1
                    break
        else:
            print(f"Epoch {ep+1}/{epochs} | Train Loss: {avg_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\nLoaded best model from epoch {best_epoch} "
            f"with Val RMSE: {best_val_rmse:.4f}"
        )

    history['best_epoch'] = best_epoch
    history['best_val_rmse'] = best_val_rmse
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    
    print("\nTraining Set:")
    print_metrics(train_metrics, prefix="  ")
    
    print("\nValidation Set:")
    print_metrics(val_metrics, prefix="  ")

    test_metrics = evaluate_model(model, test_loader, device)
    print("\nTest Set:")
    print_metrics(test_metrics, prefix="  ")

    history['final_test_metrics'] = test_metrics
    
    return model, history

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    #BASE = "/media02/tghong01/data/data"
    #BASE = "D:/KLTN/DataYelp"
    BASE = "/kaggle/input/datasets/organizations/yelp-dataset/yelp-dataset"
    print("Loading contexts...")
    user_ctx = load_user_context(f"{BASE}/yelp_academic_dataset_user.json")
    biz_ctx = load_business_context(f"{BASE}/yelp_academic_dataset_business.json")
    user_tip, biz_tip = load_tip_context(f"{BASE}/yelp_academic_dataset_tip.json")
    checkin_ctx = load_checkin_context(f"{BASE}/yelp_academic_dataset_checkin.json")
    
    dataset = YelpDataset(
        review_path=f"{BASE}/yelp_academic_dataset_review.json",
        user_ctx=user_ctx,
        biz_ctx=biz_ctx,
        user_tip=user_tip,
        biz_tip=biz_tip,
        checkin_ctx=checkin_ctx,
        max_reviews=1_000_000,
    )
    
    model, history = train_model(
        dataset,
        epochs=120,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        early_stopping_patience=30,
        early_stopping_min_delta=1e-4,
        eval_every=1  # Evaluate every epoch
    )