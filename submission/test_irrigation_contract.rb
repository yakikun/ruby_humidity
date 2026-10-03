require 'minitest/autorun'
require_relative 'irrigation_contract'
class ContractTest < Minitest::Test
  def setup
    @now=Time.iso8601('2026-10-03T12:00:00+09:00')
    @r={'device_id'=>'pot-01','observed_at'=>@now.iso8601,'soil_calibrated'=>true,'soil_moisture_pct'=>20,'raining_now'=>false}
  end
  def check(r)
    result=IrrigationContract.evaluate(r,now:@now)
    assert_equal false,result[:executable]
    assert_equal 'NO_ACTUATION',result[:command]
    assert_nil result[:recommended_water_ml]
    result
  end
  def test_candidate
    assert_equal 'watering_candidate',check(@r)[:recommendation]
  end
  def test_rain
    assert_equal 'hold',check(@r.merge('raining_now'=>true))[:recommendation]
  end
  def test_wet
    assert_equal 'hold',check(@r.merge('soil_moisture_pct'=>70))[:recommendation]
  end
  def test_invalid
    [{'soil_moisture_pct'=>Float::NAN},{'soil_moisture_pct'=>-1},{'soil_calibrated'=>false},
     {'raining_now'=>nil},{'device_id'=>''},{'observed_at'=>(@now-301).iso8601},
     {'observed_at'=>(@now+1).iso8601},{'observed_at'=>'2026-10-03T12:00:00'}].each do |bad|
      assert_equal 'review_required',check(@r.merge(bad))[:recommendation]
    end
  end
end
