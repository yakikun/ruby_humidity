# frozen_string_literal: true
require 'minitest/autorun'
require 'tmpdir'
require_relative 'training_engine'

class V2Test < Minitest::Test
  def setup
    @dir = Dir.mktmpdir('humidity_v2_test')
  end

  def teardown
    FileUtils.remove_entry(@dir)
  end

  def read_csv(text, **options)
    path = File.join(@dir, 'input.csv')
    File.write(path, text)
    HumidityData.read(path, **options)
  end

  def test_column_reordering_and_no_label_inference
    a = read_csv("sample_id,a,b\none,1,2\n", feature_names: %w[a b])
    b = read_csv("b,sample_id,a\n2,one,1\n", feature_names: %w[a b])
    assert_equal a, b
    refute a.last.first.key?(:y)
  end

  def test_bad_features
    ['', 'word', 'NaN', 'Infinity', '-Infinity', '1e999'].each do |value|
      assert_raises(ArgumentError) { read_csv("a,b\n#{value},2\n") }
    end
    ["a,a\n1,2\n", "a,b\n", "a,b\n1,2,3\n", "a,b\n1\n"].each do |text|
      assert_raises(ArgumentError) { read_csv(text) }
    end
    assert_raises(ArgumentError) { read_csv("a\n1\n", feature_names: %w[a b]) }
    assert_raises(ArgumentError) { read_csv("a,b,c\n1,2,3\n", feature_names: %w[a b]) }
    assert_raises(ArgumentError) { read_csv("sample_id,a\nx,1\nx,2\n") }
  end

  def test_bad_labels_and_split
    [-1, 101, 'NaN', ''].each do |target|
      assert_raises(ArgumentError) { read_csv("split,a,humidity_pct\ntrain,1,#{target}\n", training: true) }
    end
    assert_raises(ArgumentError) { read_csv("split,a,humidity_pct\ntest,1,50\n", training: true) }
    assert_raises(ArgumentError) { read_csv("a\n1\n", training: true) }
  end

  def test_network_validation_and_roundtrip
    m = HumidityMLP.new(2, hidden_size: 3)
    expected = m.predict_percent([0.4, -0.3])
    saved = m.state
    restored = HumidityMLP.from_state(JSON.parse(JSON.generate(saved)))
    assert_equal expected, restored.predict_percent([0.4, -0.3])
    m.train_one([0.4, -0.3], 80)
    assert_equal saved, restored.state
    assert_raises(ArgumentError) { m.forward([1]) }
    assert_raises(ArgumentError) { m.forward([Float::NAN, 1]) }
    assert_raises(ArgumentError) { m.train_one([1, 1], 101) }
    assert_raises(ArgumentError) { m.train_one([1, 1], 50, learning_rate: -1) }
    assert_raises(ArgumentError) { HumidityMLP.from_state(saved.merge(w2: [1])) }
    assert_raises(ArgumentError) { HumidityMLP.from_state(saved.merge(b2: Float::INFINITY)) }
  end

  def test_scaler_and_model_schema
    mean, std = HumidityData.fit_scaler([{ x: [2.0, 1.0] }, { x: [4.0, 1.0] }])
    assert_equal [3.0, 1.0], mean
    assert_equal [1.0, 0.01], std
    path = File.join(@dir, 'model.json')
    payload = { schema_version: 2, feature_names: %w[a b], feature_mean: mean, feature_std: std, network: HumidityMLP.new(2).state }
    File.write(path, JSON.generate(payload))
    model, = HumidityData.load_model(path)
    assert_equal 2, model.input_size
    [payload.merge(schema_version: 1), payload.merge(feature_names: %w[a a]), payload.merge(feature_std: [0, 1])].each do |bad|
      File.write(path, JSON.generate(bad))
      assert_raises(ArgumentError) { HumidityData.load_model(path) }
    end
  end

  # Deterministic mock trajectory tests checkpoint control, not learning quality.
  class Trajectory
    def initialize
      @step = 0
    end
    def train_one(*args, **options)
      @step += 1
      0.1
    end
    def state
      p = [0.6, 0.5, 0.7, 0.7, 0.8][@step - 1]
      { input_size: 1, hidden_size: 1, w1: [[0.0]], b1: [0.0], w2: [0.0], b2: Math.log(p / (1-p)) }
    end
    def predict_percent(x)
      HumidityMLP.from_state(state).predict_percent(x)
    end
  end

  def test_best_checkpoint_and_early_stopping
    rows = [{ x: [0.0], y: 50.0 }]
    result = HumidityTrainer.fit(Trajectory.new, rows, rows, epochs: 5, patience: 2)
    assert_equal 2, result[:best_epoch]
    assert_equal 4, result[:epochs_completed]
    assert_in_delta 50, result[:model].predict_percent([0]), 1e-10
    assert result[:history].all? { |h| !h.key?(:diagnostic) && h[:validation].key?(:mse_normalized) }
    full = HumidityTrainer.fit(Trajectory.new, rows, rows, epochs: 5, patience: 0)
    assert_equal 5, full[:epochs_completed]
    assert_equal 2, full[:best_epoch]
  end

  def test_learning_and_metric_units
    model = HumidityMLP.new(1)
    initial = model.loss_and_gradients([1.0], 80)[:loss]
    200.times { model.train_one([1.0], 80) }
    assert_operator model.loss_and_gradients([1.0], 80)[:loss], :<, initial
    metrics = HumidityData.metrics([50.0, 60.0], [55.0, 55.0])
    assert_equal 5, metrics[:mae_rh_points]
    assert_equal 25, metrics[:mse_rh_points_squared]
    assert_equal 0.0025, metrics[:mse_normalized]
  end

  def test_exact_validation_tie_keeps_first_epoch
    model = HumidityMLP.from_state(input_size: 1, hidden_size: 1, w1: [[0.0]], b1: [0.0], w2: [0.0], b2: 0.0)
    rows = [{ x: [0.0], y: 50.0 }]
    result = HumidityTrainer.fit(model, rows, rows, epochs: 5, patience: 2)
    assert_equal 1, result[:best_epoch]
    assert_equal 3, result[:epochs_completed]
  end
end
