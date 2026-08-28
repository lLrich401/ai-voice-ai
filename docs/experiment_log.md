# Experiment Log

- 2026-08-28 Stage A AASIST base32 synthetic 100/30, 2 epochs, VAL-A 0.724 (File 0.25, Voice 0.25, Music 0.28, Voice AUC 0.478, Music 0.676) – previous 0.813 was synthetic 16ch older
- 2026-08-28 Stage B SpecCNN Mel128+CNN14 style (beats_backbone) – music specialist, placeholder BEATs would be FMA/FakeMusicCaps real data, expected music EER 0.15
- 2026-08-28 Stage C Fusion: PANNs 0.6+AASIST 0.4 presence, DF_Arena 0.5+ AASIST 0.5 + probOR, presence-aware calibration, RMS separation check
- 2026-08-28 Integration: Added PANNs CNN14 (src/models/panns.py, fallback if no pretrained) and HTDemucs wrapper (src/models/demucs_wrapper.py, demucs->hpss)
- 2026-08-28 Inference 5 files 2-4s (DF 1.89s/file CPU), 1200 proj 38min CPU / 10min L4, silence 0.02 PASS, sample order PASS
- 2026-08-28 Submit validated: tools/validate_submission.py PASS, submit.zip includes model/df_arena 1.37GB + best.pt 2.3MB
- Next: Real data training (LibriSpeech/ASVspoof5/WaveFake/MLAAD/FMA/MusicCaps/FakeMusicCaps 100k+), PANNs pretrained Cnn14_mAP=0.431.pth (332MB) bundle, HTDemucs weights optional, full cosine AMP training 50 epochs, VAL-B/C/D (unseen/mp3/telephone) robustness
