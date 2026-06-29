rm -rf submodules/diff-ges-surfel-rasterization/third_party/glm
rm -rf submodules/diff-surfel-rasterization-original/third_party/glm

git clone https://github.com/g-truc/glm.git
cd glm
git checkout 5c46b9c07008ae65cb81ab79cd677ecc1934b903

cd ..
cp -r glm submodules/diff-ges-surfel-rasterization/third_party
cp -r glm submodules/diff-surfel-rasterization-original/third_party
cp -r glm submodules/diff_gh_surfel_rasterization/third_party

rm -rf glm