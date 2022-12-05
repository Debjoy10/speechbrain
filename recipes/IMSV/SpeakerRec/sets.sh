## Closed
# # Val
# python speaker_verification_cosine.py hparams/verification_DA_ecapa.yaml
# python speaker_verification_cosine.py hparams/verification_ecapa_trained.yaml
# python speaker_verification_cosine.py hparams/verification_DA_vggvox.yaml
# python speaker_verification_cosine.py hparams/verification_vggvox.yaml
## Open
# # Val
# python speaker_verification_cosine.py hparams/verification_DA_ecapa_pretrained.yaml
# python speaker_verification_cosine.py hparams/verification_ecapa.yaml
# python speaker_verification_cosine.py hparams/verification_ecapa_trained_pretrained.yaml

# # Test
# python speaker_verification_cosine.py hparams/verification_DA_ecapa_test.yaml
# python speaker_verification_cosine.py hparams/verification_ecapa_trained_test.yaml
# python speaker_verification_cosine.py hparams/verification_DA_vggvox_test.yaml
# python speaker_verification_cosine.py hparams/verification_vggvox_test.yaml
# # Test
# python speaker_verification_cosine.py hparams/verification_DA_ecapa_pretrained_test.yaml
# python speaker_verification_cosine.py hparams/verification_ecapa_test.yaml
# python speaker_verification_cosine.py hparams/verification_ecapa_trained_pretrained_test.yaml

# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_ecapa_big/scores.txt
# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_ecapa_big_trained/scores.txt
# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_ecapa_big_trained_pretrained/scores.txt
# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_ecapa_DA_pretrained/scores.txt
# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_vggvox_DA/scores.txt
# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_vggvox/scores.txt
# python utils/submission.py --scores_file /media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_ecapa_DA/scores.txt

# Private
# python speaker_verification_cosine.py hparams/private/verification_DA_ecapa_test.yaml
# python speaker_verification_cosine.py hparams/private/verification_ecapa_trained_test.yaml
# python speaker_verification_cosine.py hparams/private/verification_DA_ecapa_pretrained_test.yaml
# python speaker_verification_cosine.py hparams/private/verification_ecapa_test.yaml
# python speaker_verification_cosine.py hparams/private/verification_ecapa_trained_pretrained_test.yaml

CUDA_VISIBLE_DEVICES="1" python3 speaker_verification_cosine.py hparams/selftest/verification_ecapa.yaml
# CUDA_VISIBLE_DEVICES="1" python3 speaker_verification_cosine.py hparams/selftest/verification_ecapa_trained.yaml
CUDA_VISIBLE_DEVICES="1" python3 speaker_verification_cosine.py hparams/selftest/verification_ecapa_trained_pretrained.yaml
CUDA_VISIBLE_DEVICES="1" python3 speaker_verification_cosine.py hparams/selftest/verification_DA_ecapa_lossmod_0.01.yaml
