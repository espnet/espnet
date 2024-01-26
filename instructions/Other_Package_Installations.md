# 1. GIT LFS
Across all users (or when a user has sudo permissions)

```bash
cd /path/to/downloaded/folder
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar -xvfz git-lfs-linux-amd64*.tar.gz
cd git-lfs-linux-amd64*
sudo ./install.sh
```
For a specific user (or when a user does not have sudo permissions)

```bash
cd /path/to/downloaded/folder
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar -xvfz git-lfs*.tar.gz
mkdir -p ~/bin
mv git-lfs*/git-lfs ~/bin/
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cd /path/to/directory/where/git/lfs/should/be/used
git lfs install
```
