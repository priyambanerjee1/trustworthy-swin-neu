import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from src.config import Config
from src.dataset import build_dataloaders
from src.model import build_swin_t
from src.calibration import optimize_temperature, compute_ece
from src.uncertainty import extract_mc_uncertainty, normalize_with_reference
from src.decision import compute_trust_score, assign_decisions, evaluate_threshold_sweep, compute_risk_coverage

def main():
    _, val_loader, shift_loader, class_names = build_dataloaders()
    model = build_swin_t(num_classes=Config.NUM_CLASSES, dropout_p=0.2, pretrained=False).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    
    # 1. Standard In-Domain Predictions
    y_true, y_pred, raw_confs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(Config.DEVICE))
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            raw_confs.extend(confs.cpu().numpy())
            
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("--- IN-DOMAIN CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # 2. Temperature Calibration
    optimal_T = optimize_temperature(model, val_loader)
    print(f"Optimal Temperature (T): {optimal_T:.6f}")
    
    # 3. MC Dropout & Fixed Reference Extraction
    raw_u_id, mc_conf_id = extract_mc_uncertainty(model, val_loader, temp=optimal_T, n_mc=Config.MC_SAMPLES, desc="ID MC Dropout")
    U_MIN_REF, U_MAX_REF = float(raw_u_id.min()), float(raw_u_id.max())
    print(f"Locked In-Domain Bounds: U_min={U_MIN_REF:.8e}, U_max={U_MAX_REF:.8e}")
    
    # 4. In-Domain Trust & Policy
    norm_u_id = normalize_with_reference(raw_u_id, U_MIN_REF, U_MAX_REF)
    trust_id = compute_trust_score(mc_conf_id, norm_u_id)
    decisions_id = assign_decisions(trust_id)
    
    df_trust = pd.DataFrame({
        "true_label": y_true, "prediction": y_pred, "confidence": mc_conf_id,
        "uncertainty": raw_u_id, "normalized_uncertainty": norm_u_id,
        "trust_score": trust_id, "decision": decisions_id
    })
    df_trust.to_csv(os.path.join(Config.OUTPUT_DIR, "trust_results.csv"), index=False)
    
    # Threshold sweep
    df_sweep = evaluate_threshold_sweep(trust_id, y_pred, y_true)
    df_sweep.to_csv(os.path.join(Config.OUTPUT_DIR, "threshold_analysis.csv"), index=False)
    
    # 5. Distribution Shift (Gaussian Blur)
    raw_u_shift, mc_conf_shift = extract_mc_uncertainty(model, shift_loader, temp=optimal_T, n_mc=Config.MC_SAMPLES, desc="Shift MC Dropout")
    norm_u_shift = normalize_with_reference(raw_u_shift, U_MIN_REF, U_MAX_REF)
    trust_shift = compute_trust_score(mc_conf_shift, norm_u_shift)
    decisions_shift = assign_decisions(trust_shift)
    
    # True shift predictions
    shift_preds = []
    with torch.no_grad():
        for images, _ in shift_loader:
            preds = model(images.to(Config.DEVICE)).argmax(dim=1)
            shift_preds.extend(preds.cpu().numpy())
    shift_preds = np.array(shift_preds)
    
    df_shift = pd.DataFrame({
        "confidence": mc_conf_shift, "raw_uncertainty": raw_u_shift,
        "normalized_uncertainty": norm_u_shift, "trust_score": trust_shift,
        "decision": decisions_shift
    })
    df_shift.to_csv(os.path.join(Config.OUTPUT_DIR, "shift_results.csv"), index=False)
    
    # 6. Risk-Coverage Curves
    cov_c, risk_c = compute_risk_coverage(mc_conf_shift, y_true, shift_preds)
    cov_t, risk_t = compute_risk_coverage(trust_shift, y_true, shift_preds)
    aurc_c = float(np.trapezoid(risk_c, cov_c))
    aurc_t = float(np.trapezoid(risk_t, cov_t))
    print(f"Shift AURC -> Confidence-Only: {aurc_c:.6f} | Trust-Aware: {aurc_t:.6f}")
    
    pd.DataFrame({"coverage": cov_t, "risk_confidence": risk_c, "risk_trust": risk_t}).to_csv(
        os.path.join(Config.OUTPUT_DIR, "risk_coverage_shift.csv"), index=False
    )
    print("Evaluation complete. All CSV artifacts exported.")

if __name__ == "__main__":
    main()
