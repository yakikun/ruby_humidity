#!/usr/bin/env ruby
# frozen_string_literal: true
require 'json'
require_relative 'sensor_decision'

abort 'usage: ruby decide_irrigation.rb READING.json [CONFIG.json]' unless [1, 2].include?(ARGV.length)
reading = JSON.parse(File.read(ARGV[0]))
config = ARGV[1] ? JSON.parse(File.read(ARGV[1])) : {}
puts JSON.pretty_generate(SensorDecision.evaluate(reading, config))
