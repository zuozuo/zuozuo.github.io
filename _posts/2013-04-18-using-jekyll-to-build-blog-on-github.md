---
layout: post
title: 使用jekyll在github上搭建个人博客
---

无意之中看到了这个叫 jekyll 的东西，挺吸引我的，于是研究了一下，最后搭建了这个博客，第一篇就写一下自己配置搭建 jekyll 的一些经验吧。
# 介绍
# 安装
# 配置
# 发布
# 增强

    ActiveRecord::Base.connection.execute("delete from authentications where user_id not in (select id from users);")
    ActiveRecord::Base.connection.execute("delete from access_grants where user_id not in(select id from users);")
    ActiveRecord::Base.connection.execute("update users set id=id+100000;")
    ActiveRecord::Base.connection.execute("UPDATE users SET id=(@temp:=id), id = circle_id, circle_id = @temp where circle_id is not null;")
    ActiveRecord::Base.connection.execute("update users set circle_id=id-100000 where id > 100000;")
    ActiveRecord::Base.connection.execute("update users set circle_id=circle_id-100000 where circle_id > 100000;")

