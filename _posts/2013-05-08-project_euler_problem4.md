---
layout: post 
title: Project Euler Problem 4 最大回文数问题 
--- 

先来看题目：

>A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 * 99. Find the largest palindrome made from the product of two 3-digit numbers.

>回文数即是指从前向后读和从后向前读都是相同的数，已知：所有两个两位数的乘积所组成的集合中的最大回文数是： 9009 = 91 * 99；请找出所有两个三位数乘积所组成的集合中的最大回文数。 


拿到问题之后先用最直接的思路写出了一个程序：

```ruby
  def largest_palindrome_product
    [].tap {|arr| 
      100.upto(999).each do |i|
        100.upto(999).each do |j|
          prod = i*j
          prod.to_s == prod.to_s.reverse! and arr << prod
        end
      end
    }.max
  end
  p largest_palindrome_product       #output=>:     906609
```

上面的程序思路非常简单：就是从遍历所有两位数乘积然后将其中是回文数的放入数组`arr`中，然后取出回文数数组中的最大值 `arr.max`, 执行程序得出了正确答案：`906609`

但是稍微多想一下就会发现上面的程序的效率很低，两层循环，对 **i**（被乘数）和 **j**（乘数）都是从 **999** 遍历到 **100**，很显然这里面是有重复遍历的情况的：**123 * 321** 和 **321 * 123** 都会被遍历，但是我们由于他们相乘的结果是相同的，所以我们只需要遍历一边，所以优化的方法是在内层循环中，对 **j** 只遍历比当前 **i** 大的三位数。

下面优化过的程序：

```ruby
  def largest_palindrome_product_optimized
    [].tap {|arr| 
      100.upto(999).each do |i|
        i.upto(999).each do |j|
          prod = i*j
          prod.to_s == prod.to_s.reverse! and arr << prod
        end
      end
    }.max
  end
  p largest_palindrome_product_optimized       #output=>:     906609
```

但是上面的程序需要将所有的回文数放入数组中再取最大，这就增加了空间复杂度：`O(arr.length) = O(1239)` 和 时间复杂度：遍历数组取最大元素  `O(a.length) = O(1239)`  那有没有更好的方法呢？ 观察之后我们会发现我们的两层循环在遍历三位数乘积的时候同时也是对所有的回文数做了一边遍历，我们可以直接在第一次遍历的时候就取出最大元素：

```ruby
  def largest_palindrome_product_optimized1
    max = 0
    100.upto(999).each do |i|
      i.upto(999).each do |j|
        prod = i*j
        prod.to_s == prod.to_s.reverse! && prod > max and max = prod
      end
    end
    max
  end
  p largest_palindrome_product_optimized1
```

到此为止就得到了一个相对较优的算法，但是<a href="http://projecteuler.net/thread=4" target="_blank" >projecteuler.net</a>上面给出的代码里还有一个更优的方案

首先因为：` 999*999 = 998001 ` 并且**100001** 是一个回文数，这样就确定了我们所找的最大回文数一定是一个六位数。
然后有就有下面的先来看一个数学推导： 

      abccba  = a*100000 + b*10000 + c*1000 + c*100 + b*10 +a*1
              = a*(100001) + b*(10010) + c*(1100)
              = 11*(9091*a + 910*b + 100*c)

其中 **abccba** 代表一个六位的回文数，而 **a, b, c** 满足 ` 0 <= a, b, c <= 9`，这样根据上面的推导我们就能简单有效的的遍历六位数中的所有回文数，这个可以通过一个三层的循环分别从 **0..9** 遍历 **a, b, c** 来实现。  

根据推导式我们还能得出更多的信息：由于 **a, b, c** 在六位回文数中的占位是从高到低的，所以我们遍历的时候选择最高位的 **a** 为最外层循环，最低位的 **c** 为最内层循环，这样我们遍历到的第一个“符合条件”的回文数就是最大的回文数，这样就减少了遍历次数，进一步提高了效率。  

上面我们提到“符合条件”的回文数，那这个条件是指什么条件呢？ 我们的推导式只限制了遍历到的数字是回文数，但是不能保证这个数可以从两个三位数的乘积得到。

      abccba = 11*(9091*a + 910*b + 100*c)
             = 11 * x * y
             = (11*x) * y
      其中： 
        99 < 11*x, y <= 999
        9  <     x   <= 90
        99 <     y   <= 999

我们只需要验证 **abccba** 能够拆分为两个三位数 **(11*x)** 和 **y** 的乘积就可以了，所以我们需要在 **10..90** 上面遍历 **x** 只要这时候 **abccba** 能够整除 **11*x** 并且所除的的结果 **y** 满足 ` 99 < y <= 999`, 那这个回文数就是符合条件的数了。



下面是具体的实现代码：

```ruby
  def largest_palindrome_product_best
    top = 999*999
    9.downto(0) do |a|
      9.downto(0) do |b|
        9.downto(0) do |c|
          num = 9091*a + 910*b + 100*c
          num > top and next
          90.downto(10).each do |divider|
            num%divider == 0 && num/divider < 999 and return num*11
          end
        end
      end
    end
    100001
  end
  p largest_palindrome_product_best        #output=>:     906609
```

下面我们来做一个 **benchmark** 看一下上面几个算法的效率如何：

```ruby
  require 'benchmark'
  Benchmark.bmbm do |results|
    results.report("first")      { largest_palindrome_product }
    results.report("optimized")  { largest_palindrome_product_optimized }
    results.report("optimized1") { largest_palindrome_product_optimized1 }
    results.report("best")       { largest_palindrome_product_best }
  end
```

运行结果：

       Rehearsal ----------------------------------------------
       first        0.430000   0.000000   0.430000 (0.424545)
       optimized    0.200000   0.000000   0.200000 (0.209293)
       optimized1   0.220000   0.000000   0.220000 (0.212343)
       best         0.000000   0.000000   0.000000 (0.000646)
       ------------------------------------- total: 0.850000sec
       
                        user     system      total       real
       first        0.430000   0.000000   0.430000 (0.427489)
       optimized    0.210000   0.000000   0.210000 (0.211230)
       optimized1   0.210000   0.000000   0.210000 (0.210999)
       best         0.000000   0.000000   0.000000 (0.000624)

从上面`benchmark`的结果可以看出最优算法的性能要远远优于前面几个算法。




