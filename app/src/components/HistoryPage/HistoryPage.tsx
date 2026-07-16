import { CapturesTab } from '@/components/CapturesTab/CapturesTab';
import { HistoryTable } from '@/components/History/HistoryTable';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function HistoryPage() {
  return (
    <Tabs defaultValue="transcripts" className="-mx-8 flex h-full min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 items-center justify-between gap-5 border-b border-border px-8 py-5">
        <div>
          <h1 className="text-2xl font-semibold tracking-[-0.025em]">History</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Reopen transcripts and every generated voice version.
          </p>
        </div>
        <TabsList aria-label="History type">
          <TabsTrigger value="transcripts">Transcripts</TabsTrigger>
          <TabsTrigger value="voice">Voice generations</TabsTrigger>
        </TabsList>
      </header>

      <TabsContent value="transcripts" className="mt-0 min-h-0 flex-1 px-8">
        <CapturesTab />
      </TabsContent>
      <TabsContent value="voice" className="mt-0 min-h-0 flex-1 px-8 py-5">
        <HistoryTable />
      </TabsContent>
    </Tabs>
  );
}
