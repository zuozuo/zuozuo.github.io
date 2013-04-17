require 'date'
task :default => []

namespace :post do
	desc "create a new post"
	task :new do 
		`touch _posts/#{Date.today}-#{ENV['name']}.md`
		puts "create new post:  #{Date.today}-#{ENV['name']}"
	end
end
