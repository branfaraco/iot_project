import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Simulation from "./pages/Simulation";
import NotFound from "./pages/NotFound";
import LbcsMap from "./pages/LbcsMap";
import Metric from "./pages/Metric";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/simulation" element={<Simulation />} />
          {/* Routes for external windows: LBCS map and aggregated metric */}
          <Route path="/lbcs-map" element={<LbcsMap />} />
          <Route path="/metric" element={<Metric />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
