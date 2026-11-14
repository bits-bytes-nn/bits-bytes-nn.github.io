# frozen_string_literal: true
#
# Runs every unit test in this directory: `bundle exec ruby test/run_all.rb`
#
# The plugins guard their Jekyll and Liquid registrations behind `defined?`, so
# their logic loads without a site; the bundle supplies minitest and nokogiri.

Dir[File.join(__dir__, "test_*.rb")].sort.each { |f| require f }
