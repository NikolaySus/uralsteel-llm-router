cd ~
ssh-keygen -R github.com
curl -L https://api.github.com/meta | grep ssh-rsa | awk '{print "github.com", $2}' >> ~/.ssh/known_hosts
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/NikolaySus/uralsteel-llm-router.git
cd uralsteel-llm-router
