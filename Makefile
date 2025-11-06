# ==========================================
# CGYRO → NPZ → Merge (Qi,Qe,Γ) → Train 3-output NN
# Works with:
#   - parse_cgyro_phi.py               (Fortran-aware φ parser)
#   - parse_kyflux_nt_time.py          (raw float32 ky_flux; Fortran-order reshape → (2,3,2,nt,ntime))
#   - merge_ky_into_phi.py             (merges Qi,Qe,Γ into phi NPZ)
#   - phi2flux_3species.py             (trainer with checkpoints+plots)
#   - cgyro_end2end.py                 (orchestration wrapper; optional)
# ==========================================

# ---------- Paths (override as needed) ----------
export PYTHONPATH := .

# ---------- Defaults for resume & inference ----------
RESUME_MORE := 0                  # epochs to train on resume (set >0 for resume)
INFER_HZ    := 1 5 10             # horizons to use for inference
TRAIN_HZ    :=                    # optional override for training horizons on resume (leave blank to reuse saved)

DATA_DIR      ?= ./mnt/data

# ---------- Saved NN path ----------
NN_PATH := $(DATA_DIR)/bin.cgyro.nn
# Output file for inference predictions
PRED_NPZ    := $(DATA_DIR)/preds_resume_or_infer.npz

PHI_BIN       ?= $(DATA_DIR)/bin.cgyro.kxky_phi
KY_BIN        ?= $(DATA_DIR)/bin.cgyro.ky_flux

OUT_ROOT      ?= $(DATA_DIR)/myrun
LOG_DIR       ?= $(OUT_ROOT)_logs

# ---------- φ geometry (kx, ky, theta) ----------
NR            ?= 324        # radial length in φ
NTHETA        ?= 1          # poloidal/theta points in φ
NTOR          ?= 16         # toroidal/ky points in φ (for the φ file shape)

# ---------- ky-spectrum info for ky_flux ----------
NT            ?= 16         # ky/binormal points in ky_flux (try 16; if 15, override)
N_TIME        ?= 0
DKY           ?= 0.067
K0            ?= 0.0
T_INIT        ?=250   #trim transient initial time 

# ---------- Binary formats ----------
FORCE_RAW     ?= 0          # φ file: raw=0, Fortran-unformatted=1
FORCE_FORTRAN ?= 1

# ---------- Parsing/downsampling knobs for φ ----------
STRIDE_T      ?= 1
DS_R          ?= 1
DS_THETA      ?= 1
DS_TOR        ?= 1
FFT_THETA     ?= 0
FFT_TOR       ?= 0

# ---------- Training hyperparams ----------
TC            ?= 64
HORIZONS      ?= 1 2 5
EPOCHS        ?= 20
BATCH         ?= 8
LR            ?= 3e-3
DEVICE        ?= cpu

# ---------- Script names ----------
PARSE_PHI     ?= parse_cgyro_phi.py
PARSE_KY      ?= parse_kyflux_nt_time.py     # (rename parse_and_check_kyflux.py → this)
MERGE_PY      ?= merge_ky_into_phi.py
TRAIN_SCRIPT  ?= phi2flux_3species.py
END2END       ?= cgyro_end2end.py

# ---------- Outputs ----------
PHI_NPZ       := $(OUT_ROOT)_phi.npz
KY_NPZ        := $(OUT_ROOT)_kyflux.npz
PHI_FLUX_NPZ  := $(OUT_ROOT)_phi_flux.npz

# ==========================================
# Targets
# ==========================================

.PHONY: all phi ky flux train end2end report clean tinytest

all: train report

## 1) Parse φ → NPZ
phi:
	@echo ">> Parsing φ from $(PHI_BIN)"
	python $(PARSE_PHI) \
		--bin $(PHI_BIN) \
		--Nr $(NR) --Ntheta $(NTHETA) --Ntor $(NTOR) \
		--t_init $(T_INIT)\
		--force_raw $(FORCE_RAW) --force_fortran $(FORCE_FORTRAN) \
		--stride_t $(STRIDE_T) \
		--ds_r $(DS_R) --ds_theta $(DS_THETA) --ds_tor $(DS_TOR) \
		--fft_theta $(FFT_THETA) --fft_tor $(FFT_TOR) \
		--out $(PHI_NPZ)

## 2) Parse ky_flux → NPZ (Qi, Qe, Γ); n_time is auto-inferred; nt supplied
ky:
	@echo ">> Parsing ky_flux from $(KY_BIN)  (NT=$(NT),N_TIME=$(N_TIME) ,  DkY=$(DKY), k0=$(K0))" 
	python $(PARSE_KY) \
 		--bin $(KY_BIN) \
                --n_n $(NT)\
                --n_time $(N_TIME)\
		--dky $(DKY) --k0 $(K0) \
		--save_npz $(KY_NPZ) \
		--keep_ky 

## 3) Merge Qi/Qe/Γ into φ NPZ as flux[T,3]
flux: phi ky
	@echo ">> Merging ky_flux into φ NPZ"
	python $(MERGE_PY) \
		--phi_npz $(PHI_NPZ) \
		--ky_npz  $(KY_NPZ) \
		--t_init  $(T_INIT) \
		--out     $(PHI_FLUX_NPZ)

## 4) Train NN
train: flux
	@echo ">> Training model"
	mkdir -p $(LOG_DIR)
	python $(TRAIN_SCRIPT) \
		--data $(PHI_FLUX_NPZ) \
		--Tc $(TC) --horizons $(HORIZONS) \
		--epochs $(EPOCHS) \
		--batch $(BATCH) --lr $(LR) \
		--device $(DEVICE) \
		--log_dir $(LOG_DIR) \
		--save_ckpt 1 --ckpt_every 5 --plots 1

## Optional: one-shot orchestrator (calls the same scripts internally)
end2end:
	@echo ">> Running end-to-end orchestrator"
	python $(END2END) \
		--phi_bin $(PHI_BIN) --Nr $(NR) --Ntheta $(NTHETA) --Ntor $(NTOR) \
		--force_fortran $(FORCE_FORTRAN) --force_raw $(FORCE_RAW) \
		--ky_bin $(KY_BIN) --dky $(DKY) --k0 $(K0) --nt $(NT) \
		--Tc $(TC) --horizons $(HORIZONS) \
		--epochs $(EPOCHS) --batch $(BATCH) --lr $(LR) \
		--out_root $(OUT_ROOT)

## Quick sanity report
report:
	@echo "---- Artifacts ----"
	@echo "  PHI_NPZ:      $(PHI_NPZ)"
	@echo "  KY_NPZ:       $(KY_NPZ)"
	@echo "  MERGED_NPZ:   $(PHI_FLUX_NPZ)"
	@echo "  LOGS:         $(LOG_DIR)"
	@echo "    - metrics.csv"
	@echo "    - loss_curve.png"
	@echo "    - rmse_bar.png"
	@echo "    - skill_bar.png"
	@echo "    - best.pt / last.pt"

## Tiny smoke test (short run)
tinytest:
	$(MAKE) clean
	$(MAKE) phi
	python $(PARSE_KY) --bin $(KY_BIN) --out $(KY_NPZ) --dky $(DKY) --k0 $(K0) --nt $(NT)
	python $(MERGE_PY) --phi_npz $(PHI_NPZ) --ky_npz $(KY_NPZ) --out $(PHI_FLUX_NPZ)
	mkdir -p $(LOG_DIR)
	python $(TRAIN_SCRIPT) --data $(PHI_FLUX_NPZ) --Tc 32 --horizons 1 2 --epochs 3 --batch 4 --lr 3e-4 --device cpu --log_dir $(LOG_DIR) --save_ckpt 0 --plots 1
	$(MAKE) report

# ---------- Deep model knobs ----------
DEEP_LOG_DIR    ?= $(OUT_ROOT)_logs_deep
DEEP_TC         ?= 32
DEEP_HORIZONS   ?= 1 5 10
DEEP_EPOCHS     ?= 2
DEEP_BATCH      ?= 4
DEEP_LR         ?= 1e-3
DEEP_DROPOUT    ?= 0.2
DEEP_WD         ?= 1e-4
DEEP_DEVICE     ?= cpu
DEEP_PHYS_UNITS ?= 1

# Train with the deep residual + dilated TCN model
train_deep: flux
	@echo ">> Training DEEP model"
	mkdir -p $(DEEP_LOG_DIR)
	python train_with_earlystop_rmse_deep.py \
		--data $(PHI_FLUX_NPZ) \
		--Tc $(DEEP_TC) --horizons $(DEEP_HORIZONS) \
		--epochs $(DEEP_EPOCHS) --batch $(DEEP_BATCH) --lr $(DEEP_LR) \
		--device $(DEEP_DEVICE) --log_dir $(DEEP_LOG_DIR) \
		--early_stop 12 --weight_decay $(DEEP_WD) --dropout $(DEEP_DROPOUT) \
		--phys_units $(DEEP_PHYS_UNITS)

# ============================================================
# 6️⃣ Resume from saved NN and/or Infer on new horizons
# ============================================================

# Resume training for RESUME_MORE epochs (if >0), then also run inference on INFER_HZ.
# If NN_PATH doesn't exist and RESUME_MORE=0, this will error out (as intended).
resume_deep: flux
	@echo ">> Resume from $(NN_PATH) (more_epochs=$(RESUME_MORE)) and/or infer on horizons: $(INFER_HZ)"
	mkdir -p $(DEEP_LOG_DIR)_resume
	python resume_or_infer_deep.py \
		--data $(PHI_FLUX_NPZ) \
		--nn_path $(NN_PATH) \
		--Tc $(DEEP_TC) \
		--horizons $(INFER_HZ) \
		--more_epochs $(RESUME_MORE) \
		$(if $(TRAIN_HZ),--train_horizons $(TRAIN_HZ),) \
		--batch $(DEEP_BATCH) \
		--lr $(DEEP_LR) \
		--weight_decay $(DEEP_WD) \
		--dropout $(DEEP_DROPOUT) \
		--device $(DEEP_DEVICE) \
		--log_dir $(DEEP_LOG_DIR)_resume \
		--out_npz $(PRED_NPZ)

# Only inference: load NN_PATH and run forward on INFER_HZ (no training).
infer_hz: flux
	@echo ">> Inference only from $(NN_PATH) on horizons: $(INFER_HZ)"
	PYTHONPATH=. $(PYTHON) resume_or_infer_deep.py \
		--data $(PHI_FLUX_NPZ) \
		--nn_path $(NN_PATH) \
		--Tc $(DEEP_TC) \
		--horizons $(INFER_HZ) \
		--more_epochs 0 \
		--device $(DEEP_DEVICE) \
		--out_npz $(PRED_NPZ)
	@echo "Saved predictions → $(PRED_NPZ)"

# ============================================================
# 🧠 Save or rebuild unified NN file (bin.cgyro.nn) from best.pt
# ============================================================
save_nn:
	@echo ">> Rebuilding $(NN_PATH) from $(DEEP_LOG_DIR)/best.pt ..."
	@PYTHONPATH=. $(PYTHON) scripts/save_nn.py \
		--log_dir $(DEEP_LOG_DIR) \
		--nn_path $(NN_PATH) \
		--Tc $(DEEP_TC)


clean:
	rm -f $(PHI_NPZ) $(KY_NPZ) $(PHI_FLUX_NPZ)
	rm -rf $(LOG_DIR)

