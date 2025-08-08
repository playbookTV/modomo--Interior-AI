import { useState, useEffect } from 'react';
import { Brain, Settings, BarChart3, Zap, AlertCircle, CheckCircle2 } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import ObjectDetectionOverlay from './components/ObjectDetectionOverlay';
import { aiService } from './services/api';
import type { 
  RoomMakeoverResponse, 
  HealthStatus, 
  StyleComparisonResult,
  EnhancedMakeoverRequest 
} from './types/api';

const AVAILABLE_STYLES = [
  'Modern', 'Scandinavian', 'Industrial', 'Bohemian', 'Traditional', 
  'Minimalist', 'Vintage', 'Rustic', 'Contemporary'
];

export default function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'single' | 'comparison' | 'settings'>('single');
  const [makeoverResult, setMakeoverResult] = useState<RoomMakeoverResponse | null>(null);
  const [comparisonResult, setComparisonResult] = useState<StyleComparisonResult | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Single makeover settings
  const [selectedStyle, setSelectedStyle] = useState('Modern');
  const [budgetRange, setBudgetRange] = useState<'low' | 'medium' | 'high'>('medium');
  const [useEnhanced, setUseEnhanced] = useState(false);
  const [enhancedSettings, setEnhancedSettings] = useState({
    strength: 0.75,
    guidance_scale: 7.5,
    num_inference_steps: 20,
    quality_level: 'high' as const,
  });

  // Style comparison settings
  const [comparisonStyles, setComparisonStyles] = useState(['Modern', 'Scandinavian', 'Industrial']);

  // Load health status on mount
  useEffect(() => {
    const loadHealthStatus = async () => {
      try {
        const status = await aiService.getHealth();
        setHealthStatus(status);
      } catch (err) {
        console.error('Failed to load health status:', err);
      }
    };

    loadHealthStatus();
    const interval = setInterval(loadHealthStatus, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, []);

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setMakeoverResult(null);
    setComparisonResult(null);
    setError(null);
  };

  const handleImageRemove = () => {
    setSelectedImage(null);
    setMakeoverResult(null);
    setComparisonResult(null);
    setError(null);
  };

  const handleSingleMakeover = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    try {
      // For demo purposes, we'll use a mock URL
      // In production, you'd upload to storage first
      const photoUrl = `https://demo.reroom.app/uploads/${Date.now()}_${selectedImage.name}`;
      const photoId = `demo_${Date.now()}`;

      let result: RoomMakeoverResponse;

      if (useEnhanced) {
        const enhancedRequest: EnhancedMakeoverRequest = {
          photo_url: photoUrl,
          photo_id: photoId,
          style_preference: selectedStyle,
          budget_range: budgetRange,
          use_multi_controlnet: true,
          quality_level: enhancedSettings.quality_level,
          strength: enhancedSettings.strength,
          guidance_scale: enhancedSettings.guidance_scale,
          num_inference_steps: enhancedSettings.num_inference_steps,
        };

        result = await aiService.createEnhancedMakeover(enhancedRequest);
      } else {
        result = await aiService.createMakeover({
          photo_url: photoUrl,
          photo_id: photoId,
          style_preference: selectedStyle,
          budget_range: budgetRange,
        });
      }

      setMakeoverResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Makeover failed');
    } finally {
      setLoading(false);
    }
  };

  const handleStyleComparison = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    try {
      const base64 = await aiService.fileToBase64(selectedImage);
      
      const result = await aiService.createStyleComparison({
        image_data: base64,
        styles: comparisonStyles,
        include_original: true,
      });

      setComparisonResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Style comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const getHealthStatusIcon = () => {
    if (!healthStatus) return <AlertCircle className="w-5 h-5 text-gray-400" />;
    
    switch (healthStatus.status) {
      case 'healthy':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'unhealthy':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <Brain className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">LLM Demo</h1>
                <p className="text-sm text-gray-500">Interior Design AI Testing Environment</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {getHealthStatusIcon()}
              <span className="text-sm text-gray-600">
                {healthStatus?.status || 'Unknown'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200 mb-8">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('single')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'single'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Zap className="w-4 h-4 inline mr-2" />
              Single Makeover
            </button>
            <button
              onClick={() => setActiveTab('comparison')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'comparison'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Style Comparison
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'settings'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Settings className="w-4 h-4 inline mr-2" />
              Settings & Monitoring
            </button>
          </nav>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                {activeTab === 'single' ? 'Room Makeover' : 
                 activeTab === 'comparison' ? 'Style Comparison' : 'System Settings'}
              </h2>

              {/* Image Upload */}
              {activeTab !== 'settings' && (
                <div className="mb-6">
                  <ImageUpload
                    onImageSelect={handleImageSelect}
                    onImageRemove={handleImageRemove}
                    selectedImage={selectedImage}
                    disabled={loading}
                  />
                </div>
              )}

              {/* Single Makeover Controls */}
              {activeTab === 'single' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Style Preference
                    </label>
                    <select
                      value={selectedStyle}
                      onChange={(e) => setSelectedStyle(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      {AVAILABLE_STYLES.map(style => (
                        <option key={style} value={style}>{style}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Budget Range
                    </label>
                    <select
                      value={budgetRange}
                      onChange={(e) => setBudgetRange(e.target.value as 'low' | 'medium' | 'high')}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="low">Low (Under £100)</option>
                      <option value="medium">Medium (£100-£300)</option>
                      <option value="high">High (£300-£1000)</option>
                    </select>
                  </div>

                  <div>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={useEnhanced}
                        onChange={(e) => setUseEnhanced(e.target.checked)}
                        className="rounded"
                      />
                      <span className="text-sm font-medium text-gray-700">Enhanced Mode</span>
                    </label>
                  </div>

                  {useEnhanced && (
                    <div className="space-y-3 p-3 bg-gray-50 rounded-md">
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Strength: {enhancedSettings.strength}
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="1.0"
                          step="0.05"
                          value={enhancedSettings.strength}
                          onChange={(e) => setEnhancedSettings({
                            ...enhancedSettings,
                            strength: parseFloat(e.target.value)
                          })}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Guidance Scale: {enhancedSettings.guidance_scale}
                        </label>
                        <input
                          type="range"
                          min="1.0"
                          max="20.0"
                          step="0.5"
                          value={enhancedSettings.guidance_scale}
                          onChange={(e) => setEnhancedSettings({
                            ...enhancedSettings,
                            guidance_scale: parseFloat(e.target.value)
                          })}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Inference Steps: {enhancedSettings.num_inference_steps}
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="50"
                          step="5"
                          value={enhancedSettings.num_inference_steps}
                          onChange={(e) => setEnhancedSettings({
                            ...enhancedSettings,
                            num_inference_steps: parseInt(e.target.value)
                          })}
                          className="w-full"
                        />
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleSingleMakeover}
                    disabled={!selectedImage || loading}
                    className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Processing...' : 'Generate Makeover'}
                  </button>
                </div>
              )}

              {/* Style Comparison Controls */}
              {activeTab === 'comparison' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select Styles to Compare
                    </label>
                    <div className="space-y-2">
                      {AVAILABLE_STYLES.slice(0, 6).map(style => (
                        <label key={style} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={comparisonStyles.includes(style)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setComparisonStyles([...comparisonStyles, style]);
                              } else {
                                setComparisonStyles(comparisonStyles.filter(s => s !== style));
                              }
                            }}
                            className="rounded"
                          />
                          <span className="text-sm text-gray-700">{style}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={handleStyleComparison}
                    disabled={!selectedImage || loading || comparisonStyles.length === 0}
                    className="w-full py-2 px-4 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Comparing...' : 'Compare Styles'}
                  </button>
                </div>
              )}

              {/* System Settings */}
              {activeTab === 'settings' && (
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 rounded-md">
                    <h3 className="font-medium text-gray-900 mb-2">System Health</h3>
                    <div className="space-y-2">
                      {healthStatus?.components && Object.entries(healthStatus.components).map(([name, component]) => (
                        <div key={name} className="flex items-center justify-between text-sm">
                          <span className="capitalize">{name.replace('_', ' ')}</span>
                          <span className={`px-2 py-1 rounded text-xs ${
                            component.status === 'healthy' ? 'bg-green-100 text-green-800' :
                            component.status === 'degraded' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {component.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <button
                    onClick={async () => {
                      try {
                        await aiService.clearCache();
                        alert('Cache cleared successfully');
                      } catch (err) {
                        alert('Failed to clear cache');
                      }
                    }}
                    className="w-full py-2 px-4 bg-red-600 text-white rounded-md hover:bg-red-700"
                  >
                    Clear Redis Cache
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
                <div className="flex">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">Error</h3>
                    <p className="mt-1 text-sm text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {loading && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">
                  {activeTab === 'single' ? 'Generating makeover...' : 'Comparing styles...'}
                </p>
                <p className="text-sm text-gray-500 mt-2">This may take 15-30 seconds</p>
              </div>
            )}

            {!loading && makeoverResult?.transformation && activeTab === 'single' && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Makeover Result</h3>
                
                <ObjectDetectionOverlay
                  imageUrl={makeoverResult.transformation.after_image_url}
                  detectedObjects={makeoverResult.transformation.detected_objects}
                  suggestedProducts={makeoverResult.transformation.suggested_products}
                  imageWidth={800}
                  imageHeight={600}
                />

                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-md">
                    <h4 className="font-medium text-gray-900 mb-2">Processing Details</h4>
                    <p className="text-sm text-gray-600">
                      Processing Time: {makeoverResult.processing_time_ms}ms
                    </p>
                    <p className="text-sm text-gray-600">
                      Style: {makeoverResult.transformation.style_name}
                    </p>
                    <p className="text-sm text-gray-600">
                      Objects Detected: {makeoverResult.transformation.detected_objects.length}
                    </p>
                  </div>
                  
                  <div className="p-4 bg-gray-50 rounded-md">
                    <h4 className="font-medium text-gray-900 mb-2">Cost Estimate</h4>
                    <p className="text-sm text-gray-600">
                      Total: £{makeoverResult.transformation.total_estimated_cost}
                    </p>
                    <p className="text-sm text-gray-600">
                      Savings: £{makeoverResult.transformation.savings_amount}
                    </p>
                    <p className="text-sm text-gray-600">
                      Products: {makeoverResult.transformation.suggested_products.length}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {!loading && comparisonResult && activeTab === 'comparison' && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Style Comparison</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(comparisonResult.results).map(([key, result]) => (
                    <div key={key} className="border border-gray-200 rounded-md p-3">
                      <h4 className="font-medium text-gray-900 mb-2 capitalize">{result.style}</h4>
                      
                      {result.image_data && (
                        <img
                          src={result.image_data}
                          alt={`${result.style} style`}
                          className="w-full h-48 object-cover rounded mb-2"
                        />
                      )}
                      
                      {result.error ? (
                        <p className="text-red-600 text-sm">{result.error}</p>
                      ) : result.suggested_products && (
                        <div className="text-sm">
                          <p className="font-medium text-gray-700 mb-1">Top Products:</p>
                          {result.suggested_products.slice(0, 2).map((product, idx) => (
                            <p key={idx} className="text-gray-600 text-xs">
                              • {product.name} ({product.price_range})
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && !makeoverResult && !comparisonResult && activeTab !== 'settings' && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
                <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for AI Analysis</h3>
                <p className="text-gray-600">
                  Upload a room image and select your preferences to get started with AI-powered interior design analysis.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}