'use client';

import { useState, useEffect } from 'react';
import Select, { MultiValue, ActionMeta } from 'react-select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Типы для данных API
interface Position {
  ticker: string;
  shares: number;
}

interface TickerOption {
  value: string;
  label: string;
}

interface Recommendation {
  ticker: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  expected_return: number;
  reasoning: string;
  current_price?: number;
  target_price?: number;
}

interface PortfolioResponse {
  recommendations: Recommendation[];
  total_value?: number;
  analysis_timestamp?: string;
  model_version?: string;
}

// Компонент карточки рекомендации
function RecommendationCard({ rec }: { rec: Recommendation }) {
  const getActionStyles = (action: string) => {
    switch (action) {
      case 'BUY':
        return { badge: 'action-buy', icon: '🟢', label: 'ПОКУПАТЬ' };
      case 'SELL':
        return { badge: 'action-sell', icon: '🔴', label: 'ПРОДАВАТЬ' };
      default:
        return { badge: 'action-hold', icon: '🟡', label: 'ДЕРЖАТЬ' };
    }
  };

  const styles = getActionStyles(rec.action);
  
  const getConfidenceClass = (confidence: number) => {
    if (confidence >= 0.7) return 'confidence-high';
    if (confidence >= 0.5) return 'confidence-medium';
    return 'confidence-low';
  };

  return (
    <div className="recommendation-card">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold text-gray-900">{rec.ticker}</h3>
          {rec.current_price && (
            <p className="text-gray-600">Цена: ${rec.current_price.toFixed(2)}</p>
          )}
        </div>
        <span className={`action-badge ${styles.badge}`}>
          {styles.icon} {styles.label}
        </span>
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600">Уверенность</span>
          <span className="font-semibold">{(rec.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="confidence-bar">
          <div 
            className={`confidence-fill ${getConfidenceClass(rec.confidence)}`}
            style={{ width: `${rec.confidence * 100}%` }}
          />
        </div>
      </div>

      <div className="mb-4">
        <p className="text-sm text-gray-600 mb-1">Ожидаемая доходность:</p>
        <p className={`text-lg font-semibold ${rec.expected_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {rec.expected_return >= 0 ? '+' : ''}{rec.expected_return.toFixed(2)}%
        </p>
      </div>

      <div className="bg-gray-50 rounded p-3">
        <p className="text-sm text-gray-700">
          <span className="font-semibold">Обоснование:</span> {rec.reasoning}
        </p>
      </div>
    </div>
  );
}

// Компонент формы портфеля с выпадающим списком тикеров
function PortfolioForm({ 
  onSubmit, 
  isLoading,
  availableTickers
}: { 
  onSubmit: (data: { cash: number; positions: Position[] }) => void;
  isLoading: boolean;
  availableTickers: TickerOption[];
}) {
  const [cash, setCash] = useState<number>(10000);
  const [positions, setPositions] = useState<Position[]>([
    { ticker: '', shares: 0 }
  ]);

  const addPosition = () => {
    setPositions([...positions, { ticker: '', shares: 0 }]);
  };

  const removePosition = (index: number) => {
    setPositions(positions.filter((_, i) => i !== index));
  };

  const updatePosition = (index: number, field: keyof Position, value: string | number) => {
    const updated = [...positions];
    updated[index] = { ...updated[index], [field]: value };
    setPositions(updated);
  };

  const handleTickerChange = (index: number, selectedOption: TickerOption | null) => {
    updatePosition(index, 'ticker', selectedOption?.value || '');
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validPositions = positions.filter(p => p.ticker.trim() !== '');
    onSubmit({ cash, positions: validPositions });
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md p-6 mb-8">
      <h2 className="text-xl font-bold mb-4 text-gray-900">Параметры портфеля</h2>
      
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Доступная наличность ($)
        </label>
        <input
          type="number"
          value={cash}
          onChange={(e) => setCash(parseFloat(e.target.value) || 0)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          min="0"
          step="100"
        />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Позиции в портфеле
        </label>
        
        {positions.map((pos, index) => (
          <div key={index} className="flex gap-2 mb-2">
            <Select
              options={availableTickers}
              value={availableTickers.find(opt => opt.value === pos.ticker) || null}
              onChange={(option) => handleTickerChange(index, option as TickerOption | null)}
              placeholder="Выберите тикер..."
              className="flex-1"
              classNamePrefix="react-select"
              isSearchable={true}
              isClearable={true}
              noOptionsMessage={() => "Нет доступных тикеров"}
              styles={{
                control: (base) => ({
                  ...base,
                  borderColor: '#d1d5db',
                  '&:hover': { borderColor: '#9ca3af' }
                })
              }}
            />
            <input
              type="number"
              placeholder="Кол-во"
              value={pos.shares || ''}
              onChange={(e) => updatePosition(index, 'shares', parseInt(e.target.value) || 0)}
              className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
              min="0"
            />
            {positions.length > 1 && (
              <button
                type="button"
                onClick={() => removePosition(index)}
                className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-md"
              >
                ✕
              </button>
            )}
          </div>
        ))}

        <button
          type="button"
          onClick={addPosition}
          className="mt-2 text-sm text-primary-600 hover:text-primary-700 font-medium"
        >
          + Добавить позицию
        </button>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-primary-600 text-white py-3 px-4 rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-colors"
      >
        {isLoading ? 'Анализ...' : 'Анализировать портфель'}
      </button>
    </form>
  );
}

// Главный компонент страницы
export default function Home() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalValue, setTotalValue] = useState<number | null>(null);
  const [modelVersion, setModelVersion] = useState<string>('');
  const [availableTickers, setAvailableTickers] = useState<TickerOption[]>([]);

  // Загрузка списка доступных тикеров при монтировании компонента
  useEffect(() => {
    const fetchTickers = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/tickers');
        if (response.ok) {
          const data = await response.json();
          const options = data.tickers.map((ticker: string) => ({
            value: ticker,
            label: ticker
          }));
          setAvailableTickers(options);
        } else {
          // Fallback список тикеров если API недоступно
          const fallbackTickers = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'YNDX', 'TCSG', 'VTBR', 'ROSN', 'GMKN', 'NLMK'];
          setAvailableTickers(fallbackTickers.map(ticker => ({ value: ticker, label: ticker })));
        }
      } catch (err) {
        console.error('Ошибка загрузки тикеров:', err);
        // Fallback список тикеров
        const fallbackTickers = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'YNDX', 'TCSG', 'VTBR', 'ROSN', 'GMKN', 'NLMK'];
        setAvailableTickers(fallbackTickers.map(ticker => ({ value: ticker, label: ticker })));
      }
    };

    fetchTickers();
  }, []);

  const handleAnalyze = async (data: { cash: number; positions: Position[] }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ошибка при анализе портфеля');
      }

      const result: PortfolioResponse = await response.json();
      setRecommendations(result.recommendations);
      setTotalValue(result.total_value || null);
      setModelVersion(result.model_version || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Произошла неизвестная ошибка');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            📈 Investment Advisor
          </h1>
          <p className="text-gray-600">
            Система инвестиционных рекомендаций на основе машинного обучения
          </p>
          {modelVersion && (
            <p className="text-sm text-gray-500 mt-2">
              Версия модели: {modelVersion}
            </p>
          )}
        </header>

        {/* Form */}
        <PortfolioForm 
          onSubmit={handleAnalyze} 
          isLoading={isLoading}
          availableTickers={availableTickers}
        />

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            <strong>Ошибка:</strong> {error}
          </div>
        )}

        {/* Results */}
        {recommendations.length > 0 && (
          <div>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Рекомендации по портфелю
              </h2>
              {totalValue && (
                <div className="text-right">
                  <p className="text-sm text-gray-600">Общая стоимость портфеля</p>
                  <p className="text-2xl font-bold text-primary-600">
                    ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </p>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map((rec) => (
                <RecommendationCard key={rec.ticker} rec={rec} />
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && recommendations.length === 0 && !error && (
          <div className="text-center py-12 text-gray-500">
            <p className="text-lg">Введите данные портфеля и нажмите &quot;Анализировать&quot;</p>
            <p className="text-sm mt-2">
              Система проанализирует технические индикаторы, сентимент новостей и выдаст рекомендации
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
