# frozen_string_literal: true
require 'optparse'
require 'digest'
require 'fileutils'
require_relative 'data_io'
require_relative 'training_engine'

options = { epochs: 80, learning_rate: 0.03, seed: 42, patience: 10 }
parser = OptionParser.new do |o|
  o.banner = 'ruby train.rb --data CSV --output NEW_DIR [--patience 0]'
  o.on('--data PATH') { |v| options[:data] = v }
  o.on('--output PATH') { |v| options[:output] = v }
  o.on('--epochs N', Integer) { |v| options[:epochs] = v }
  o.on('--learning-rate N', Float) { |v| options[:learning_rate] = v }
  o.on('--seed N', Integer) { |v| options[:seed] = v }
  o.on('--patience N', Integer) { |v| options[:patience] = v }
end
parser.parse!
raise ArgumentError, parser.to_s unless options[:data] && options[:output] && ARGV.empty?
raise ArgumentError, 'output exists; use a new directory' if File.exist?(options[:output])
names, rows = HumidityData.read(options[:data], training: true)
groups = %w[train validation diagnostic].map { |s| rows.select { |r| r[:split] == s } }
raise ArgumentError, 'all3splits required' if groups.any?(&:empty?)
mean, std = HumidityData.fit_scaler(groups[0])
train, validation, diagnostic = groups.map { |g| HumidityData.transform(g, mean, std) }
source_hash = Digest::SHA256.file(options[:data]).hexdigest
FileUtils.mkdir_p(options[:output])
out = options[:output]
File.write(File.join(out, 'protocol.json'), JSON.pretty_generate(options.merge(feature_names: names, input_sha256: source_hash,
  selection: 'minimum validation MAE; earliest epoch on exact tie', diagnostic_used_for_selection: false,
  counts: groups.map(&:length), feature_statistics: 'training split only', optimizer: 'online SGD', loss: 'MSE of RH/100')))
model = HumidityMLP.new(names.length, hidden_size: 32, seed: options[:seed])
result = HumidityTrainer.fit(model, train, validation, **options.select { |k, _| %i[epochs learning_rate seed patience].include?(k) })
best = result[:model]
payload = { schema_version: 2, feature_names: names, feature_mean: mean, feature_std: std,
            network: best.state, best_epoch: result[:best_epoch], input_sha256: source_hash,
            note: 'Feature-vector regressor, not a Ruby ResNet. Labels must be interpreted using the dataset provenance.' }
File.write(File.join(out, 'model.json'), JSON.pretty_generate(payload))
File.write(File.join(out, 'last_model.json'), JSON.pretty_generate(payload.merge(network: result[:last_state], best_epoch: nil, epoch: result[:epochs_completed])))
File.write(File.join(out, 'history.json'), JSON.pretty_generate(result[:history]))
restored, = HumidityData.load_model(File.join(out, 'model.json'))
predictions = diagnostic.map { |r| { id: r[:id], reference: r[:y], prediction: best.predict_percent(r[:x]) } }
reload_error = diagnostic.zip(predictions).map { |r, p| (restored.predict_percent(r[:x]) - p[:prediction]).abs }.max
raise 'save/load mismatch' unless reload_error < 1e-10
report = { best_epoch: result[:best_epoch], epochs_completed: result[:epochs_completed],
           train: HumidityTrainer.evaluate(best, train), validation: HumidityTrainer.evaluate(best, validation),
           diagnostic: HumidityTrainer.evaluate(best, diagnostic),
           last_epoch_diagnostic: HumidityTrainer.evaluate(model, diagnostic),
           reload_max_error_rh_points: reload_error, input_sha256: source_hash }
raise 'input changed during training' unless Digest::SHA256.file(options[:data]).hexdigest == source_hash
File.write(File.join(out, 'metrics.json'), JSON.pretty_generate(report))
File.write(File.join(out, 'diagnostic_predictions.json'), JSON.pretty_generate(predictions))
puts JSON.pretty_generate(report)
