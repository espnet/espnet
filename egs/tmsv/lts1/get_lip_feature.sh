nj=16        # numebr of parallel jobs
shape_predictor_path=downloads/shape_predictor_68_face_landmarks.dat
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
echo "stage 2: Face Feature Generation"

face_feature_dir=face_feature_for_gg

make_face.sh --cmd "${train_cmd}" --nj ${nj} \
            --fps 60 \
            --lip_width 128 \
            --lip_height 64 \
            --shape_predictor_path ${shape_predictor_path} \
            ${face_feature_dir}\
            ${face_feature_dir}\
            ${face_feature_dir}

