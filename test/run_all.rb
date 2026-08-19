# frozen_string_literal: true
#
# Runs every unit test in this directory: `ruby test/run_all.rb`
#
# Deliberately plain `ruby`, not `bundle exec`. The plugins guard their Jekyll and
# Liquid registrations behind `defined?`, so their logic loads standalone, and
# minitest ships with Ruby — which keeps the suite runnable without the bundle
# and without adding a test gem to the Gemfile.

Dir[File.join(__dir__, "test_*.rb")].sort.each { |f| require f }
