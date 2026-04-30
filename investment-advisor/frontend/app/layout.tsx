import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Investment Advisor - Система инвестиционных рекомендаций',
  description: 'ML-система для анализа портфеля и генерации рекомендаций по акциям',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ru">
      <body className="antialiased bg-gray-50 min-h-screen">{children}</body>
    </html>
  )
}
