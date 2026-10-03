require 'json'
require 'fileutils'
require_relative '../data_io'
require_relative 'irrigation_contract'

root = File.expand_path('..', __dir__)
out = ARGV.fetch(0) { abort 'usage: ruby submission/demo.rb NEW_OUTPUT_DIR' }
abort 'output exists; choose a new directory' if File.exist?(out)
model, payload = HumidityData.load_model(File.join(root, 'runs/controlled_s42/model.json'))
names, rows = HumidityData.read(File.join(root, 'data/v6_compact_features.csv'), training: true)
row = rows.find { |r| r[:split] == 'diagnostic' }
x = HumidityData.transform([row], payload['feature_mean'], payload['feature_std']).first[:x]
prediction = {sample_id: row[:id], predicted_future_proxy_humidity_pct: model.predict_percent(x),
              reference_proxy_humidity_pct: row[:y], input: '16 precomputed image-summary features',
              school_local_truth: false, new_unseen_test: false}
now = Time.now.utc
sensor = {'device_id'=>'demo-pot-01','observed_at'=>now.iso8601,
          'soil_moisture_pct'=>27.0,'soil_calibrated'=>true,'raining_now'=>false}
decision = IrrigationContract.evaluate(sensor, now: now)
FileUtils.mkdir_p(out)
{'prediction'=>prediction,'synthetic_sensor_input'=>sensor,'device_dry_run'=>decision}.each do |name, value|
  File.write(File.join(out, name+'.json'), JSON.pretty_generate(value))
end
puts JSON.pretty_generate(prediction: prediction, device_dry_run: decision,
  note: 'Real archived feature prediction plus separate SYNTHETIC sensor demonstration. Humidity is not converted to rain probability. No hardware access.')
