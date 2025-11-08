"use client"

import * as React from "react"
import { Suspense } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/lib/auth-context"
import { Sparkles, LogIn } from "lucide-react"

function LoginForm() {
  const [email, setEmail] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [isLoading, setIsLoading] = React.useState(false)
  const [showSignup, setShowSignup] = React.useState(false)
  const [name, setName] = React.useState("")
  const { login, signup, loginAsGuest } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()

  // Auto-login as guest if guest param is present
  React.useEffect(() => {
    if (searchParams.get("guest") === "true") {
      loginAsGuest()
      router.push("/dashboard")
    }
  }, [searchParams, loginAsGuest, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    try {
      if (showSignup) {
        await signup(email, password, name)
      } else {
        await login(email, password)
      }
      router.push("/dashboard")
    } catch (error) {
      console.error("Auth error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleGuestLogin = () => {
    loginAsGuest()
    router.push("/dashboard")
  }

  return (
    <div className="min-h-screen gradient-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="glass rounded-3xl p-8 md:p-10 shadow-2xl">
          {/* Logo/Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-md mb-4">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">MirrorMe</h1>
            <p className="text-white/80">Your AI Communication Coach</p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mb-6 p-1 bg-white/10 rounded-xl backdrop-blur-md">
            <button
              onClick={() => setShowSignup(false)}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all ${
                !showSignup
                  ? "bg-white/20 text-white shadow-lg"
                  : "text-white/70 hover:text-white"
              }`}
            >
              Sign In
            </button>
            <button
              onClick={() => setShowSignup(true)}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all ${
                showSignup
                  ? "bg-white/20 text-white shadow-lg"
                  : "text-white/70 hover:text-white"
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {showSignup && (
              <div className="space-y-2">
                <Label htmlFor="name" className="text-white/90">
                  Full Name
                </Label>
                <Input
                  id="name"
                  type="text"
                  placeholder="John Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required={showSignup}
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email" className="text-white/90">
                Email
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-white/90">
                Password
              </Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
              />
            </div>

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full h-12 bg-white/20 hover:bg-white/30 text-white border border-white/30 backdrop-blur-md font-semibold text-base shadow-lg transition-all"
            >
              {isLoading ? (
                "Loading..."
              ) : showSignup ? (
                "Create Account"
              ) : (
                <>
                  <LogIn className="w-4 h-4 mr-2" />
                  Sign In
                </>
              )}
            </Button>
          </form>

          {/* Guest Login */}
          <div className="mt-6 pt-6 border-t border-white/20">
            <Button
              onClick={handleGuestLogin}
              variant="ghost"
              className="w-full h-12 text-white/90 hover:text-white hover:bg-white/10 backdrop-blur-md border border-white/20 rounded-xl transition-all"
            >
              Continue as Guest
            </Button>
            <p className="text-xs text-white/60 text-center mt-3">
              Guest mode allows you to try MirrorMe without creating an account
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    }>
      <LoginForm />
    </Suspense>
  )
}

