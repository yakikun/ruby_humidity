# frozen_string_literal: true
require 'json'

class HumidityMLP
  attr_reader :input_size, :hidden_size

  def initialize(input_size, hidden_size: 32, seed: 42)
    raise ArgumentError, 'positive integer dimensions required' unless [input_size, hidden_size].all? { |n| n.is_a?(Integer) && n.positive? }
    @input_size, @hidden_size = input_size, hidden_size
    rng = Random.new(seed)
    scale1 = Math.sqrt(6.0 / (input_size + hidden_size))
    scale2 = Math.sqrt(6.0 / (hidden_size + 1))
    @w1 = Array.new(hidden_size) { Array.new(input_size) { (rng.rand * 2 - 1) * scale1 } }
    @b1 = Array.new(hidden_size, 0.0)
    @w2 = Array.new(hidden_size) { (rng.rand * 2 - 1) * scale2 }
    @b2 = 0.0
  end

  def self.finite_number?(v)
    v.is_a?(Numeric) && v.finite?
  end

  def sigmoid(x)
    # Stable evaluation without clipping the activation or its derivative.
    x >= 0 ? 1.0 / (1.0 + Math.exp(-x)) : (e = Math.exp(x); e / (1.0 + e))
  end

  def forward(x)
    unless x.is_a?(Array) && x.length == @input_size && x.all? { |v| self.class.finite_number?(v) }
      raise ArgumentError, "expected #{@input_size} finite features"
    end
    z = @w1.map.with_index { |row, j| row.zip(x).sum { |w, v| w * v } + @b1[j] }
    hidden = z.map { |v| [v, 0.0].max }
    logit = @w2.zip(hidden).sum { |w, v| w * v } + @b2
    raise ArgumentError, 'numeric overflow' unless (z + [logit]).all?(&:finite?)
    { z1: z, hidden: hidden, probability: sigmoid(logit) }
  end

  def predict_percent(x)
    forward(x)[:probability] * 100.0
  end

  def loss_and_gradients(x, target_percent)
    unless self.class.finite_number?(target_percent) && target_percent.between?(0, 100)
      raise ArgumentError, 'humidity must be finite and between0and100'
    end
    c = forward(x)
    error = c[:probability] - target_percent / 100.0
    dz2 = 2.0 * error * c[:probability] * (1.0 - c[:probability])
    dz1 = @w2.each_index.map { |j| c[:z1][j] > 0 ? @w2[j] * dz2 : 0.0 }
    gradients = { w1: dz1.map { |d| x.map { |v| d * v } }, b1: dz1,
                  w2: c[:hidden].map { |v| dz2 * v }, b2: dz2 }
    { prediction_percent: c[:probability] * 100.0, loss: error**2, gradients: gradients }
  end

  def train_one(x, target_percent, learning_rate: 0.03)
    raise ArgumentError, 'positive finite learning rate required' unless self.class.finite_number?(learning_rate) && learning_rate.positive?
    r = loss_and_gradients(x, target_percent)
    g = r[:gradients]
    @w1.each_index do |j|
      @w1[j].each_index { |i| @w1[j][i] -= learning_rate * g[:w1][j][i] }
      @b1[j] -= learning_rate * g[:b1][j]
      @w2[j] -= learning_rate * g[:w2][j]
    end
    @b2 -= learning_rate * g[:b2]
    r[:loss]
  end

  def state
    # Return a copy: best-epoch checkpoints must not follow later updates.
    { input_size: @input_size, hidden_size: @hidden_size, w1: @w1.map(&:dup),
      b1: @b1.dup, w2: @w2.dup, b2: @b2 }
  end

  def self.from_state(input)
    s = input.transform_keys(&:to_sym)
    model = new(s.fetch(:input_size), hidden_size: s.fetch(:hidden_size), seed: 0)
    h, n = model.hidden_size, model.input_size
    valid = s[:w1].is_a?(Array) && s[:w1].length == h && s[:w1].all? { |row| row.is_a?(Array) && row.length == n } &&
            s[:b1].is_a?(Array) && s[:b1].length == h && s[:w2].is_a?(Array) && s[:w2].length == h
    raise ArgumentError, 'invalid weight shape' unless valid
    raise ArgumentError, 'nonfinite weights' unless [s[:w1], s[:b1], s[:w2], s[:b2]].flatten.all? { |v| finite_number?(v) }
    model.instance_variable_set(:@w1, s[:w1].map(&:dup))
    model.instance_variable_set(:@b1, s[:b1].dup)
    model.instance_variable_set(:@w2, s[:w2].dup)
    model.instance_variable_set(:@b2, s[:b2])
    model
  end
end
