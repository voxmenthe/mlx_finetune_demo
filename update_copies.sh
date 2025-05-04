cd ~/repos/OTHER_PEOPLES_REPOS

cd mlx-lm && git pull
cd ..
cp -r mlx-lm ~/repos/mlx_overall/

cd mlx-examples && git pull
cd ..
cp -r mlx-examples ~/repos/mlx_overall/

cd mlx-pretrain && git pull
cd ..
cp -r mlx-pretrain ~/repos/mlx_overall/

cd mlx-vlm && git pull
cd ..
cp -r mlx-vlm ~/repos/mlx_overall/

cd mlx-audio && git pull
cd ..
cp -r mlx-audio ~/repos/mlx_overall/

cd synthetic-data-kit && git pull
cd ..
cp -r synthetic-data-kit ~/repos/mlx_overall/

cd ~/repos/mlx_overall/
cd mlx-examples && rm -rf .git*
cd ../mlx-lm && rm -rf .git*
cd ../mlx-pretrain && rm -rf .git*
cd ../mlx-vlm && rm -rf .git*
cd ../mlx-audio && rm -rf .git*
cd ../synthetic-data-kit && rm -rf .git*

echo "Done!"







