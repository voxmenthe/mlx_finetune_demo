rm -rf .git*
git init
git add .
git commit -m "Initial commit"
gh repo create --public --source=. --push
git status
