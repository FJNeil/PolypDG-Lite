import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  metadataBase: new URL("https://polypdg-lite-research.fjff.chatgpt.site"),
  title: "PolypDG-Lite | Cross-Center Polyp Segmentation",
  description: "A deployment-oriented framework combining six-center LOCO evaluation, consistency learning, knowledge distillation, and low-power FP16 validation.",
  openGraph: {
    title: "PolypDG-Lite",
    description: "Cross-center robustness. Low-power deployment.",
    images: [{ url: "/og.png", width: 1680, height: 945, alt: "PolypDG-Lite cross-center research framework" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "PolypDG-Lite",
    description: "Cross-center robustness. Low-power deployment.",
    images: ["/og.png"],
  },
};
export default function RootLayout({children}: Readonly<{children:React.ReactNode}>) { return <html lang="en"><body>{children}</body></html>; }
