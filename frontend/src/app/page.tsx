"use client"

import { useRouter } from "next/navigation"
import { useEffect } from "react"
import { useAuth } from "@/lib/auth-context"
import { Button } from "@/components/ui/button"
import { Sparkles, Video, TrendingUp, Shield, Zap } from "lucide-react"

export default function Home() {
  const router = useRouter()
  const { user } = useAuth()

  return (
    <div className="min-h-screen gradient-bg flex flex-col">
      {/* Navigation */}
      <nav className="glass border-b border-white/20">
        <div className="container mx-auto px-4 py-4 flex items-center justify-center">
          <div className="flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-white" />
            <span className="text-xl font-bold text-white">MirrorMe</span>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex items-center justify-center px-4 py-20">
        <div className="max-w-4xl text-center space-y-8">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-white/20 backdrop-blur-md mb-4">
            <Sparkles className="w-10 h-10 text-white" />
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-white leading-tight">
            Transform Your
            <span className="block bg-gradient-to-r from-yellow-300 to-pink-300 bg-clip-text text-transparent">
              Public Speaking
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-white/90 max-w-2xl mx-auto leading-relaxed">
            Real-time AI feedback on your communication skills. Practice confidently, 
            improve continuously—all privately on your device.
          </p>

          <div className="flex justify-center pt-4">
            <Button
              onClick={() => router.push("/login")}
              size="lg"
              className="h-14 px-8 bg-white/20 hover:bg-white/30 text-white border border-white/30 backdrop-blur-md text-lg font-semibold shadow-xl"
            >
              <Video className="w-5 h-5 mr-2" />
              Start Practicing
            </Button>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-6 mt-16 pt-16 border-t border-white/20">
            <div className="glass rounded-2xl p-6 backdrop-blur-md">
              <Shield className="w-10 h-10 text-white mb-4 mx-auto" />
              <h3 className="text-xl font-semibold text-white mb-2">Privacy First</h3>
              <p className="text-white/80">
                Everything runs on your device. No cloud, no data sharing.
              </p>
            </div>
            <div className="glass rounded-2xl p-6 backdrop-blur-md">
              <Zap className="w-10 h-10 text-white mb-4 mx-auto" />
              <h3 className="text-xl font-semibold text-white mb-2">Real-Time Feedback</h3>
              <p className="text-white/80">
                Get instant insights on posture, gestures, and vocal delivery.
              </p>
            </div>
            <div className="glass rounded-2xl p-6 backdrop-blur-md">
              <TrendingUp className="w-10 h-10 text-white mb-4 mx-auto" />
              <h3 className="text-xl font-semibold text-white mb-2">Track Progress</h3>
              <p className="text-white/80">
                See your confidence scores improve with each practice session.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

