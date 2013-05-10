#!/usr/bin/env ruby
# encoding=utf-8
# problem 4 on http://www/projecteuler.net

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
p largest_palindrome_product

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
p largest_palindrome_product_optimized

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

def largest_palindrome_product_optimized2
  i = 999
  max = 0
  while i > 99
    j = 999
    while j >= i
      prod = i*j
      prod.to_s == prod.to_s.reverse! && prod > max and max = prod
      j -= 1
    end
    i -= 1
  end
  max
end

# 0 <= a,b,c <= 9
# abccba = 100000*a + 10000*b + 1000*c + 100 *c + 10*b +a
#        = 100001*a + 10010*b + 1100*c
#        = 11*(9091*a + 910*b + 100*c)

p 999*999
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
p largest_palindrome_product_best


TESTS = 10_000
require 'benchmark'
Benchmark.bmbm do |results|
  results.report("first") { largest_palindrome_product }
  results.report("optimized") { largest_palindrome_product_optimized }
  results.report("optimized1") { largest_palindrome_product_optimized1 }
  results.report("best") { largest_palindrome_product_best }
end
