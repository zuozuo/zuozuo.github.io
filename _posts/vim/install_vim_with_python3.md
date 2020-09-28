## install macvim
https://github.com/macvim-dev/homebrew-macvim

## install vim by brew
```shell
brew install vim
alias vim=/usr/local/bin/vim
vim --version | grep python
```

## config vim to use python3
```vim
set pythonthreedll=/usr/local/Frameworks/Python.framework/Versions/Current/Python
```
https://github.com/macvim-dev/macvim/issues/866
https://github.com/macvim-dev/macvim/issues/1012
