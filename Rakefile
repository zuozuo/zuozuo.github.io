require 'date'
task :default => []

namespace :post do
	desc "create a new post"
	task :new do 
		post = "_posts/#{Date.today}-#{ENV['name']}.md"
		`touch #{post}`
		`echo '---\nlayout: post\ntitle: \n---\n' >> #{post}`
		puts "create new post:  #{Date.today}-#{ENV['name']}"
	end
end
